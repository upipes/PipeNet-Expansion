import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import utility.data_preparation as data_preparation
import utility.model_bases as model
import numpy as np
import torchvision
import os

def classifier_cache_path(opt, seed):
    source_dataset = opt.zstfrom if opt.zst else opt.dataset
    missing_rate = getattr(opt, "current_train_sample_missing_rate", 0)
    suffix = f"_miss{missing_rate}" if missing_rate else ""
    tag = f"_{opt.cache_tag}" if getattr(opt, "cache_tag", "") else ""
    return f'{opt.rootpath}/models/base-classifiers/{source_dataset}_{opt.image_embedding}_seed{seed}_clr{opt.classifier_lr}_nep{opt.classifier_nepoch}{suffix}{tag}'

class REGRESSOR:
    def __init__(self, _train_X, _train_Y, data_loader, _nclass, _cuda, seedinfo, train_base=False, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, _embed_dim=1000, _num_layers=3, opt=None):
        self.train_X = _train_X
        self.train_Y = _train_Y

        self.test_seen_feature = data_loader.test_seen_feature
        self.test_seen_label = data_loader.test_seen_label
        self.test_unseen_feature = data_loader.test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label

        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses

        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.embed_dim = _embed_dim
        self.num_layers = _num_layers
        self.input_dim = _train_X.size(1)
        self.cuda = _cuda
        self.model = model.LINEAR(self.input_dim, len(self.seenclasses))
        self.model.apply(data_preparation.weights_init)
        self.criterion = nn.CrossEntropyLoss()

        self.opt = opt
        self.seedinfo = seedinfo

        if self.opt.conclude_inv:
            self.attribute_f = data_loader.attribute_f
            self.attribute_inv = data_loader.attribute_inv
            self.attribute_new = data_loader.attribute_new
        else:
            self.attribute_new = data_loader.attribute_f

        self.input = torch.FloatTensor(_batch_size, self.input_dim)
        self.label = torch.LongTensor(_batch_size)

        self.lr = _lr
        self.beta1 = _beta1
        self.optimizer_classifier = optim.Adam(self.model.parameters(), lr=self.opt.classifier_lr, betas=(self.opt.classifier_beta1, 0.999))
        self.calc_entropy = False

        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]

        if not train_base:
            missing_rate = getattr(self.opt, "current_train_sample_missing_rate", 0)
            suffix = f"_miss{missing_rate}" if missing_rate else ""
            if hasattr(self.opt, "current_base_classifier_model"):
                best_model = self.opt.current_base_classifier_model
            elif self.opt.zst:
                if self.opt.zstfrom == 'imagenet':
                    source_model = torchvision.models.resnet101(pretrained=True)
                    best_model = model.LINEAR(self.test_seen_feature.size(1), len(self.seenclasses))
                    best_model.fc.weight.data[:, :] = source_model.fc.weight
                    best_model.fc.bias.data[:] = source_model.fc.bias
                    if self.cuda:
                        best_model.cuda()
                else:
                    best_model = torch.load(classifier_cache_path(self.opt, self.seedinfo))
            else:
                best_model = torch.load(classifier_cache_path(self.opt, self.seedinfo))

            self.model = best_model
            # weight + bias
            self.target_weights = torch.cat((best_model.fc.weight.data, torch.unsqueeze(best_model.fc.bias.data, 1)), 1)
            self.ref_norm = torch.norm(self.target_weights, dim=-1).mean()

    def pred_weights_and_val(self, weight_model, daegnn=None):
        """ Predict weights and insert in extended GZSL model and/or ZSL model. Then evaluate performance. """
        attributes_to_regress = self.attribute_new[self.unseenclasses]
        for n, attribute_vector in enumerate(attributes_to_regress):
            attribute_vector = attribute_vector.cuda()[None, :]
            if self.opt.concatenation:
                attribute_vector_inv = self.attribute_inv[self.unseenclasses][n].cuda()[None, :]
                if self.opt.single_autoencoder_baseline:
                    pred_weights = weight_model(attribute_vector, attribute_vector_inv, True).squeeze()
                else:
                    pred_weights = weight_model.predict(attribute_vector, attribute_vector_inv, True).squeeze()
            else:
                if self.opt.single_autoencoder_baseline:
                    pred_weights = weight_model(attribute_vector).squeeze()
                else:
                    pred_weights = weight_model.predict(attribute_vector).squeeze()

            # class injection
            self.unseen_model.fc.weight.data[n,:] = pred_weights[:self.input_dim]
            self.unseen_model.fc.bias.data[n] = pred_weights[self.input_dim]
            self.ext_model.fc.weight.data[len(self.seenclasses)+n, :] = pred_weights[:self.input_dim]
            self.ext_model.fc.bias.data[len(self.seenclasses)+n] = pred_weights[self.input_dim]

        # baseline wDAE
        if daegnn:
            self.ref_weights = torch.cat((self.ext_model.fc.weight.data, torch.unsqueeze(self.ext_model.fc.bias.data, 1)), 1).unsqueeze(0)
            pred_weights = daegnn(self.ref_weights).squeeze()
            self.unseen_model.fc.weight.data[:, :] = pred_weights[len(self.seenclasses):, :self.input_dim]
            self.unseen_model.fc.bias.data[:] = pred_weights[len(self.seenclasses):, self.input_dim]
            self.ext_model.fc.weight.data[:, :] = pred_weights[:, :self.input_dim] 
            self.ext_model.fc.bias.data[:] = pred_weights[:, self.input_dim]
        
        if self.opt.zst:
            acc_target, acc_zst_unseen = self.val_zst()
            return acc_target, acc_zst_unseen
        else:
            acc_gzsl, acc_seen, acc_unseen, H, acc_unseen_zsl = self.val_gzsl()
            return acc_gzsl, acc_seen, acc_unseen, H, acc_unseen_zsl

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_total = 0
        acc_per_class = []
        prediction_matrix = torch.zeros(
            (len(target_classes), len(target_classes)))
        for n, i in enumerate(target_classes):
            idx = (test_label == i)
            if self.opt.save_pred_matrix:
                for k, j in enumerate(target_classes):
                    prediction_matrix[n, k] = torch.sum(
                        ((predicted_label[idx]) == j)) / torch.sum(idx)
            acc = torch.sum(test_label[idx] ==
                            predicted_label[idx]) / torch.sum(idx)
            acc_per_class.append(acc)
            acc_total += acc
        acc_total /= target_classes.size(0)
        acc_per_class = torch.stack(acc_per_class)
        return acc_total, acc_per_class, prediction_matrix

    def val_model(self, model, test_X, test_label, target_classes, calc_entropy=False):
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for layer in model.children():
            if hasattr(layer, 'out_features'):
                num_out = layer.out_features

        all_outputs = torch.Tensor(ntest, len(torch.unique(target_classes)))
        all_outputs = torch.Tensor(ntest, num_out)
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                output = model(Variable(test_X[start:end].cuda()))
            else:
                output = model(Variable(test_X[start:end]))
            if calc_entropy:
                all_outputs[start:end] = output.data
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end
        acc, acc_per_class, prediction_matrix = self.compute_per_class_acc_gzsl(
            test_label, predicted_label, target_classes)
        if self.opt.save_pred_matrix:
            torch.save(acc_per_class, f'{self.opt.rootpath}/outputs/percls_acc_{self.opt.dataset}res101_finetuned_len_test_{len(test_X)}_len_tar_{len(target_classes)}.pt')
            np.savetxt(f'{self.opt.rootpath}/outputs/percls_acc_{self.opt.dataset}res101_finetuned_len_test_{len(test_X)}_len_tar_{len(target_classes)}.txt', acc_per_class.detach().cpu().numpy(), fmt="%.6f")
            torch.save(prediction_matrix, f'{self.opt.rootpath}/outputs/pred_matrix{self.opt.dataset}{self.opt.method}res101_finetuned_len_test_' + str(len(test_X)) + '_len_tar_' + str(len(target_classes)) + '.pt')
            np.savetxt(f'{self.opt.rootpath}/outputs/pred_matrix{self.opt.dataset}{self.opt.method}res101_finetuned_len_test_' + str(len(test_X)) + '_len_tar_' + str(len(target_classes)) + '.txt', prediction_matrix.detach().cpu().numpy(), fmt="%.6f")
            np.savetxt(f'{self.opt.rootpath}/outputs/pred_matrix_{self.opt.dataset}_{self.seedinfo}.txt', prediction_matrix.detach().cpu().numpy(), fmt="%.6f")
        if calc_entropy:
            from torch.distributions import Categorical
            sm = torch.nn.Softmax(dim=1)
            mean_entropy = Categorical(probs=sm(all_outputs)).entropy().mean()
            print(
                "Mean entropy (log e) of output distributions over test samples: ", mean_entropy)
        return acc

    def val_gzsl(self):      
        if self.opt.norm_scale_heuristic:
            pred_weights = torch.cat((self.unseen_model.fc.weight.data, torch.unsqueeze(self.unseen_model.fc.bias.data, 1)), 1)
            pred_weights = pred_weights / (10*torch.norm(pred_weights, dim=1).mean()) 
            self.unseen_model.fc.weight.data[:, :] = pred_weights[:, :self.input_dim]
            self.unseen_model.fc.bias.data[:] = pred_weights[:, self.input_dim]
            self.ext_model.fc.weight.data[len(self.seenclasses):, :] = pred_weights[:, :self.input_dim] 
            self.ext_model.fc.bias.data[len(self.seenclasses):] = pred_weights[:, self.input_dim]  
        
        acc_gzsl = self.val_model(self.ext_model, torch.cat((self.test_seen_feature, self.test_unseen_feature), 0),
                    torch.cat((data_preparation.map_label(self.test_seen_label, self.seenclasses), data_preparation.map_label_extend(self.test_unseen_label, self.unseenclasses, self.seenclasses)), 0),
                    torch.cat((data_preparation.map_label(self.seenclasses, self.seenclasses), data_preparation.map_label_extend(self.unseenclasses, self.unseenclasses, self.seenclasses)), 0),
                    calc_entropy=self.calc_entropy)
        acc_seen = self.val_model(self.ext_model, self.test_seen_feature, data_preparation.map_label(self.test_seen_label, self.seenclasses), data_preparation.map_label(self.seenclasses, self.seenclasses))
        acc_unseen = self.val_model(self.ext_model, self.test_unseen_feature, data_preparation.map_label_extend(self.test_unseen_label, self.unseenclasses, self.seenclasses), data_preparation.map_label_extend(self.unseenclasses, self.unseenclasses, self.seenclasses))    
        H = 2*acc_seen*acc_unseen / (acc_seen+acc_unseen)
        # ZSL 
        acc_unseen_zsl = self.val_model(self.unseen_model, self.test_unseen_feature, data_preparation.map_label(self.test_unseen_label, self.unseenclasses), data_preparation.map_label(self.unseenclasses, self.unseenclasses))    
        
        if self.opt.daegnn:
            self.ref_weights = self.ref_weights.squeeze()
            self.ext_model.fc.weight.data[:, :] = self.ref_weights[:, :self.input_dim] 
            self.ext_model.fc.bias.data[:] = self.ref_weights[:, self.input_dim]

        return acc_gzsl, acc_seen, acc_unseen, H, acc_unseen_zsl
    
    def val_zst(self):
        # resnet = torchvision.models.resnet101(pretrained=True)

        # if self.opt.norm_scale_heuristic:
        #     self.unseen_model.fc.weight.data[:, :] *= torch.norm(resnet.fc.weight.data[:, :], dim=1).mean() / torch.norm(self.unseen_model.fc.weight.data[:, :], dim=1).mean()
        #     self.unseen_model.fc.bias.data[:] *= torch.norm(resnet.fc.bias.data[:]) / torch.norm(self.unseen_model.fc.bias.data[:])
            
        # Save model for concatenation with PreTrained ResNet for ImageNet inference in eval_imagenet.py (ZST GZSL)
        if not os.path.exists(self.opt.rootpath + '/models/zst-models/'):
            os.makedirs(self.opt.rootpath + '/models/zst-models/')
        torch.save(self.unseen_model.state_dict(), self.opt.rootpath + '/models/zst-models/' + f"{self.opt.dataset}_seed{self.seedinfo}_normalized{self.opt.norm_scale_heuristic}")

        acc_target = self.val_model(self.unseen_model, self.test_unseen_feature, data_preparation.map_label(self.test_unseen_label, self.unseenclasses-len(self.seenclasses)), data_preparation.map_label(self.unseenclasses-len(self.seenclasses), self.unseenclasses-len(self.seenclasses)))
        
        # Append predicted classifier to Resnet
        # self.ext_model.fc.weight = nn.Parameter(torch.cat((resnet.fc.weight.cuda(), self.unseen_model.fc.weight)))
        # self.ext_model.fc.bias = nn.Parameter(torch.cat((resnet.fc.bias.cuda(), self.unseen_model.fc.bias)))
        # acc_zst_unseen = self.val_model(self.ext_model, self.test_unseen_feature, data_preparation.map_label(self.test_unseen_label, self.unseenclasses-len(self.seenclasses)) + len(self.seenclasses), data_preparation.map_label_extend(self.unseenclasses, self.unseenclasses, self.seenclasses))    

        return acc_target, 0
