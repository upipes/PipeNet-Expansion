import numpy as np
import scipy.io as sio
import torch
from sklearn import preprocessing
from torch.utils.data import Dataset
from utility.utils import *

from torch import Tensor
import torch.nn as nn
import re
import pandas as pd
import sys

class cos_sim_loss(nn.MSELoss):
    __constants__ = ['reduction']

    def __init__(self, dim=1, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(cos_sim_loss, self).__init__(size_average, reduce, reduction)
        assert reduction in ['none', None, 'mean', 'sum']
        self.reduction = reduction
        self.cos = nn.CosineSimilarity(dim=dim, eps=1e-8)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = 1-self.cos(input, target)
        if self.reduction == 'none' or self.reduction == None:
            return loss.unsqueeze(-1)
        elif self.reduction == 'mean':
            return loss.mean()
        else:
            return loss.sum()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i

    return mapped_label

def map_label_extend(label, new_classes, base_classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(new_classes.size(0)):
        mapped_label[label == new_classes[i]] = i + len(base_classes)
    return mapped_label


def reverse_map_label(label, classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(classes.size(0)):
        mapped_label[label == i] = classes[i]
    return mapped_label


def reverse_map_label_extend(label, new_classes, base_classes):
    mapped_label = torch.LongTensor(label.size())
    for i in range(new_classes.size(0)):
        mapped_label[label == (i + len(base_classes))] = new_classes[i]
    return mapped_label

class GenericDataset(Dataset):
    def __init__(self, opt, _input, _target, cuda, transform=None):
        assert len(_input) == len(_target)
        self.opt = opt
        self.input = _input
        self.target = _target
        self.transform = transform
        self.cuda = cuda

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        in_var = self.input[idx]
        target = self.target[idx]

        if self.cuda:
            in_var = in_var.cuda()
            target = target.cuda()

        if self.transform:
            in_var = self.transform(in_var)

        return in_var, target

class GenericDatasetINV(Dataset):
    def __init__(self, opt, _input, _target, _input_inv, _target_inv, cuda, transform=None):
        assert len(_input) == len(_target)
        self.opt = opt
        self.input = _input
        self.target = _target
        self.input_inv = _input_inv
        self.target_inv = _target_inv
        self.transform = transform
        self.cuda = cuda

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        in_var = self.input[idx]
        target = self.target[idx]
        in_var_inv = self.input_inv[idx]
        target_inv = self.target_inv[idx]

        if self.cuda:
            in_var = in_var.cuda()
            target = target.cuda()
            in_var_inv = in_var_inv.cuda()
            target_inv = target_inv.cuda()

        if self.transform:
            in_var = self.transform(in_var)

        return in_var, target, in_var_inv, target_inv

class Logger(object):
    def __init__(self, filename):
        self.filename = filename
        f = open(self.filename+'.log', "a")
        f.close()

    def write(self, message):
        f = open(self.filename+'.log', "a")
        f.write(message)
        f.close()


def get_mean_features(features, labels):
    mean_features = []
    features = np.array(features)
    labels = np.array(labels)
    for label in range(50):
        indices = np.where(labels == label)[0]
        mean_feature = np.mean(features[indices], axis=0)
        mean_features.append(mean_feature)
    mean_features = np.array(mean_features)
    return mean_features


class DATA_LOADER(object):
    def __init__(self, opt):
        if opt.matdataset:
            self.read_matdataset(opt)
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def select_random_view_subset(self, opt, attribute):
        if not opt.random_view_subset:
            return attribute
        if attribute.dim() != 3:
            raise ValueError("--random_view_subset expects embeddings with shape (class_num, global+views, emb_dim).")

        total_views = attribute.shape[1] - 1
        if opt.view_num < 0 or opt.view_num > total_views:
            raise ValueError(f"--view_num must be in [0, {total_views}] when sampling views, got {opt.view_num}.")

        if not hasattr(self, "selected_embedding_indices"):
            view_indices = np.random.choice(np.arange(1, total_views + 1), size=opt.view_num, replace=False)
            view_indices = np.sort(view_indices)
            self.selected_embedding_indices = np.concatenate(([0], view_indices)).astype(int)
            opt.selected_view_indices = ",".join(str(x) for x in self.selected_embedding_indices.tolist())
            print(f"Selected embedding indices (0 is global): {opt.selected_view_indices}")

        return attribute[:, self.selected_embedding_indices, :]

    def random_derangement(self, n):
        if n < 2:
            raise ValueError("Class permutation replacement requires at least two classes.")
        while True:
            perm = np.random.permutation(n)
            if np.all(perm != np.arange(n)):
                return perm

    def apply_class_permute_error_views(self, opt, attribute):
        if not opt.class_permute_error_views:
            return attribute
        if attribute.dim() != 3:
            raise ValueError("--class_permute_error_views expects embeddings with shape (class_num, global+views, emb_dim).")

        total_views = attribute.shape[1] - 1
        if opt.view_error_percent < 0 or opt.view_error_percent > 100:
            raise ValueError(f"--view_error_percent must be in [0, 100], got {opt.view_error_percent}.")
        error_view_num = int(round(total_views * opt.view_error_percent / 100.0))
        if error_view_num == 0:
            opt.error_view_indices = ""
            return attribute

        if not hasattr(self, "error_embedding_indices"):
            error_indices = np.random.choice(np.arange(1, total_views + 1), size=error_view_num, replace=False)
            self.error_embedding_indices = np.sort(error_indices).astype(int)
            if hasattr(self, "selected_embedding_indices"):
                original_error_indices = self.selected_embedding_indices[self.error_embedding_indices]
            else:
                original_error_indices = self.error_embedding_indices
            opt.error_view_indices = ",".join(str(x) for x in original_error_indices.tolist())
            print(f"Class-permuted error view indices (0 is global, never corrupted): {opt.error_view_indices}")

        corrupted = attribute.clone()
        for view_idx in self.error_embedding_indices:
            perm = self.random_derangement(corrupted.shape[0])
            corrupted[:, view_idx, :] = corrupted[torch.from_numpy(perm).long(), view_idx, :]
        return corrupted

    def subsample_train_per_class(self, train_feature, train_label, missing_rate):
        if missing_rate == 0:
            self.train_sample_indices = torch.arange(train_label.size(0))
            self.train_sample_counts = {
                int(cls): int((train_label == cls).sum().item())
                for cls in torch.unique(train_label).tolist()
            }
            return train_feature, train_label

        if missing_rate < 0 or missing_rate >= 100:
            raise ValueError(f"--train_sample_missing_rate must be in [0, 99], got {missing_rate}.")

        keep_rate = (100 - missing_rate) / 100.0
        sampled_indices = []
        sample_counts = {}
        for cls in torch.unique(train_label).sort().values:
            cls_indices = torch.where(train_label == cls)[0]
            keep_num = max(1, int(round(cls_indices.numel() * keep_rate)))
            perm = torch.randperm(cls_indices.numel())[:keep_num]
            picked = cls_indices[perm]
            sampled_indices.append(picked)
            sample_counts[int(cls.item())] = int(keep_num)

        sampled_indices = torch.cat(sampled_indices)
        sampled_indices = sampled_indices[torch.randperm(sampled_indices.numel())]
        self.train_sample_indices = sampled_indices.detach().cpu().clone()
        self.train_sample_counts = sample_counts
        print(f"Source training samples after {missing_rate}% per-class removal: {sample_counts}")
        return train_feature[sampled_indices], train_label[sampled_indices]

    def read_matdataset(self, opt):
        if opt.zst:
            print(
                f"Transfer setting: target_dataset={opt.dataset}, source_dataset={opt.zstfrom}; "
                "train_sample_missing_rate is applied only to the source training split."
            )
        matcontent = sio.loadmat(f"{opt.dataroot}/{opt.dataset}/{opt.image_embedding}.mat")

        # visual feature
        feature = matcontent['features']
        label = matcontent['labels'].astype(int).squeeze() - 1
        # (930, 2048)
        # print(feature.shape)

        # transfer
        if opt.zst:
            if opt.factual_branch == 'attention':
                embedding_path = f"{opt.rootpath}/embeddings/{opt.class_embedding}/{opt.zstfrom}_{opt.llm}_{opt.class_embedding}.npy"
                embedding_path_target = f"{opt.rootpath}/embeddings/{opt.class_embedding}/{opt.dataset}_{opt.llm}_{opt.class_embedding}.npy"
                self.attribute = torch.from_numpy(np.load(embedding_path, allow_pickle=True)).float()
                self.attribute_target = torch.from_numpy(np.load(embedding_path_target, allow_pickle=True)).float()
                opt.wordemb_dim = self.attribute.shape[-1]
                self.attribute = self.attribute / self.attribute.norm(dim=-1, keepdim=True)
                self.attribute_target = self.attribute_target / self.attribute_target.norm(dim=-1, keepdim=True)
                self.attribute = self.select_random_view_subset(opt, self.attribute)
                self.attribute_target = self.select_random_view_subset(opt, self.attribute_target)
                self.attribute = self.apply_class_permute_error_views(opt, self.attribute)
                self.attribute_target = self.apply_class_permute_error_views(opt, self.attribute_target)
                self.attribute = torch.cat([self.attribute, self.attribute_target], dim=0)
                # (10,11,512)
                print(self.attribute.shape) 
                self.attribute = self.attribute.reshape(self.attribute.shape[0], -1)
            self.attribute_f = self.attribute
            
        
        # not transfer
        else:
            opt.wordemb_dim = 4096
            embedding_path = f"{opt.rootpath}/embeddings/{opt.class_embedding}/{opt.dataset}_{opt.llm}_{opt.class_embedding}.npy"

            if opt.class_embedding == "clip":
                opt.wordemb_dim = 512
            elif opt.class_embedding == "sbert":
                opt.wordemb_dim = 768
            elif opt.class_embedding == "llama-8b":
                opt.wordemb_dim = 4096
            elif opt.class_embedding == "qwen-7b":
                opt.wordemb_dim = 3584
            else:
                raise ValueError("Invalid embedding model")

            # LLM-generation factual-intervention representation
            # ablation factual
            if opt.factual_branch == 'mean':
                self.attribute = torch.from_numpy(np.load(embedding_path, allow_pickle=True)).float()
                self.attribute = self.select_random_view_subset(opt, self.attribute)
                self.attribute = self.apply_class_permute_error_views(opt, self.attribute)
                self.attribute /= torch.norm(self.attribute, dim=1)[:, None]
                self.attribute = self.attribute.mean(dim=1)

            # multi-view factual
            elif opt.factual_branch == 'attention':
                self.attribute = torch.from_numpy(np.load(embedding_path, allow_pickle=True)).float()
                self.attribute = self.attribute / self.attribute.norm(dim=-1, keepdim=True)
                self.attribute = self.select_random_view_subset(opt, self.attribute)
                self.attribute = self.apply_class_permute_error_views(opt, self.attribute)
                self.attribute = self.attribute.reshape(self.attribute.shape[0], -1)
            
            self.attribute_f = self.attribute

        print(f"Loaded factual attribute shape: {self.attribute_f.shape}")
        
        # feature and label statistics
        # trainval_loc = torch.arange(len(label_number_list)-100)
        # test_seen_loc = torch.arange(len(label_number_list)-100, len(label_number_list))
        # test_unseen_loc = torch.arange(len(label_number_list))

        # seen_loc = [index for index, value in enumerate(label_number_list) if value in [0,1,2]]
        # trainval_loc = seen_loc[:-100]
        # test_seen_loc = seen_loc[-100:]
        # test_unseen_loc = [index for index, value in enumerate(label_number_list) if value in [3,4]]

        # self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
        # self.train_label = torch.tensor([label_number_list[idx] for idx in trainval_loc]).long()
        # self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
        # self.test_unseen_label = torch.tensor([label_number_list[idx] for idx in test_unseen_loc]).long()
        # self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float()
        # self.test_seen_label = torch.tensor([label_number_list[idx] for idx in test_seen_loc]).long()

        test_unseen_loc = torch.arange(len(label))
        self.test_unseen_feature = torch.from_numpy(feature[test_unseen_loc]).float()
        self.test_unseen_label = torch.tensor([label[idx] for idx in test_unseen_loc]).long()
        
        matcontent = sio.loadmat(f"{opt.dataroot}/{opt.zstfrom}/{opt.image_embedding}.mat")

        # visual feature
        feature = matcontent['features']
        label = matcontent['labels'].astype(int).squeeze() - 1


        trainval_loc = torch.arange(len(label)-50)
        test_seen_loc = torch.arange(len(label)-50, len(label))
        # torch.Size([752, 2048])
        self.train_feature = torch.from_numpy(feature[trainval_loc]).float()
        self.train_label = torch.tensor([label[idx] for idx in trainval_loc]).long()
        self.test_seen_feature = torch.from_numpy(feature[test_seen_loc]).float()
        self.test_seen_label = torch.tensor([label[idx] for idx in test_seen_loc]).long()

        missing_rate = getattr(
            opt,
            "current_train_sample_missing_rate",
            getattr(opt, "train_sample_missing_rate", [0])[0],
        )
        self.train_feature, self.train_label = self.subsample_train_per_class(
            self.train_feature,
            self.train_label,
            missing_rate,
        )
        self.source_missing_rate_applied = missing_rate

        self.seenclasses = torch.from_numpy(np.unique(self.train_label.numpy()))
        self.unseenclasses = torch.from_numpy(np.unique(self.test_unseen_label.numpy()))

        if opt.zst:
            self.unseenclasses = self.unseenclasses + len(self.seenclasses)
            self.seenclasses = torch.arange(len(self.seenclasses))

        print(self.unseenclasses)
        print(self.seenclasses)

        self.nclass = len(self.seenclasses) + len(self.unseenclasses)
        self.ntrain = self.train_feature.size()[0]
        self.train_mapped_label = map_label(self.train_label, self.seenclasses)
