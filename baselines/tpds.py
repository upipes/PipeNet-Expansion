import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import trange

import utility.data_preparation as data_preparation
from regressor import REGRESSOR


class TPDSNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        logits = self.classifier(feat)
        return feat, logits

    def predict(self, x):
        _, logits = self.forward(x)
        return logits


class TPDS(REGRESSOR):
    def __init__(self, opt, **kwargs):
        super().__init__(train_base=True, opt=opt, **kwargs)
        self.opt = opt

        self.model = TPDSNet(
            input_dim=self.input_dim,
            hidden_dim=self.embed_dim if isinstance(self.embed_dim, int) else self.opt.dann_hidden_dim,
            num_classes=len(self.seenclasses),
        )
        self.model.apply(data_preparation.weights_init)
        self.class_criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, betas=(self.beta1, 0.999)
        )

        if self.cuda:
            self.model.cuda()
            self.class_criterion.cuda()

        self.target_feature = self.test_unseen_feature
        self._pretrain_source()
        self._adapt_target()
        self.evaluate()

    def _pretrain_source(self):
        best_state = None
        best_score = float("-inf")
        source_size = self.train_X.size(0)

        for _ in trange(self.opt.classifier_nepoch):
            self.model.train()
            perm = torch.randperm(source_size)
            for start in range(0, source_size, self.batch_size):
                end = min(source_size, start + self.batch_size)
                idx = perm[start:end]
                source_x = self.train_X[idx]
                source_y = self.train_Y[idx]
                if self.cuda:
                    source_x = source_x.cuda()
                    source_y = source_y.cuda()

                self.optimizer.zero_grad()
                _, logits = self.model(source_x)
                loss = self.class_criterion(logits, source_y)
                loss.backward()
                self.optimizer.step()

            score = self._source_validation_score()
            if score > best_score:
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }

        if best_state is not None:
            self.model.load_state_dict(best_state)
            if self.cuda:
                self.model.cuda()

    def _adapt_target(self):
        best_state = None
        best_score = float("-inf")

        for _ in trange(self.nepoch):
            self.model.train()
            target_features, target_probs, proxy_probs = self._build_proxy_distribution()
            perm = torch.randperm(target_features.size(0))

            for start in range(0, target_features.size(0), self.batch_size):
                end = min(target_features.size(0), start + self.batch_size)
                idx = perm[start:end]
                batch_x = target_features[idx]
                batch_probs = target_probs[idx]
                batch_proxy = proxy_probs[idx]

                if self.cuda:
                    batch_x = batch_x.cuda()
                    batch_probs = batch_probs.cuda()
                    batch_proxy = batch_proxy.cuda()

                feat, logits = self.model(batch_x)
                pred = F.softmax(logits, dim=1)

                align_loss = F.kl_div(torch.log(pred + 1e-5), batch_proxy, reduction="batchmean")
                cc_loss = F.mse_loss(pred, batch_probs)
                entropy_loss = torch.mean(torch.sum(-pred * torch.log(pred + 1e-5), dim=1))
                mean_pred = pred.mean(dim=0)
                gentropy_loss = torch.sum(-mean_pred * torch.log(mean_pred + 1e-5))
                im_loss = entropy_loss - gentropy_loss

                loss = (
                    self.opt.tpds_align_weight * align_loss
                    + self.opt.tpds_cc_weight * cc_loss
                    + self.opt.tpds_im_weight * im_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            score = self._target_validation_score()
            if score > best_score:
                best_score = score
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }

        if best_state is not None:
            self.model.load_state_dict(best_state)
            if self.cuda:
                self.model.cuda()

    def _build_proxy_distribution(self):
        self.model.eval()
        with torch.no_grad():
            features = []
            probs = []
            for start in range(0, self.target_feature.size(0), self.batch_size):
                end = min(self.target_feature.size(0), start + self.batch_size)
                batch = self.target_feature[start:end]
                if self.cuda:
                    batch = batch.cuda()
                feat, logits = self.model(batch)
                features.append(F.normalize(feat, dim=1).detach().cpu())
                probs.append(F.softmax(logits, dim=1).detach().cpu())

            features = torch.cat(features, dim=0)
            probs = torch.cat(probs, dim=0)
            proxy = probs.clone()

            for _ in range(self.opt.tpds_steps):
                distances = torch.cdist(features, features)
                knn_idx = torch.topk(
                    distances,
                    k=min(self.opt.tpds_neighbors + 1, distances.size(1)),
                    largest=False,
                    dim=1,
                ).indices[:, 1:]
                gathered = proxy[knn_idx]
                neighbor_proxy = gathered.mean(dim=1)
                proxy = 0.5 * proxy + 0.5 * neighbor_proxy
                proxy = proxy / (proxy.sum(dim=1, keepdim=True) + 1e-5)

            return self.target_feature, probs, proxy

    def evaluate(self):
        if self.opt.zst:
            target_classes = self.unseenclasses - len(self.seenclasses)
            self.acc_target = self.eval_with_class_subset(
                self.test_unseen_feature,
                data_preparation.map_label(self.test_unseen_label, target_classes),
                data_preparation.map_label(target_classes, target_classes),
                class_indices=target_classes,
                save_tag="tpds_target_zsl",
                calc_entropy=self.calc_entropy,
            )
            self.acc_unseen_zsl = self.acc_target

            self.acc_zst_unseen = self.eval_with_class_subset(
                self.test_unseen_feature,
                self.test_unseen_label,
                self.seenclasses,
                class_indices=self.seenclasses,
                save_tag="tpds_target_gzsl",
            )
            self.acc_gzsl = self.acc_zst_unseen
            self.acc_seen = self.acc_zst_unseen.new_zeros(())
            self.acc_unseen = self.acc_zst_unseen
            self.H = self.acc_zst_unseen.new_zeros(())
        else:
            gzsl_features = torch.cat((self.test_seen_feature, self.test_unseen_feature), dim=0)
            gzsl_labels = torch.cat(
                (
                    data_preparation.map_label(self.test_seen_label, self.seenclasses),
                    data_preparation.map_label_extend(
                        self.test_unseen_label, self.unseenclasses, self.seenclasses
                    ),
                ),
                dim=0,
            )
            gzsl_target_classes = torch.cat(
                (
                    data_preparation.map_label(self.seenclasses, self.seenclasses),
                    data_preparation.map_label_extend(
                        self.unseenclasses, self.unseenclasses, self.seenclasses
                    ),
                ),
                dim=0,
            )

            self.acc_gzsl = self.eval_full_space(
                gzsl_features,
                gzsl_labels,
                gzsl_target_classes,
                save_tag="tpds_gzsl",
                calc_entropy=self.calc_entropy,
            )
            self.acc_seen = self.eval_full_space(
                self.test_seen_feature,
                data_preparation.map_label(self.test_seen_label, self.seenclasses),
                data_preparation.map_label(self.seenclasses, self.seenclasses),
                save_tag="tpds_seen",
            )
            self.acc_unseen = self.eval_full_space(
                self.test_unseen_feature,
                data_preparation.map_label_extend(
                    self.test_unseen_label, self.unseenclasses, self.seenclasses
                ),
                data_preparation.map_label_extend(
                    self.unseenclasses, self.unseenclasses, self.seenclasses
                ),
                save_tag="tpds_unseen",
            )
            denom = self.acc_seen + self.acc_unseen
            self.H = 2 * self.acc_seen * self.acc_unseen / denom if denom.item() > 0 else denom
            self.acc_unseen_zsl = torch.zeros((), device=self.acc_unseen.device, dtype=self.acc_unseen.dtype)

    def _source_validation_score(self):
        self.model.eval()
        with torch.no_grad():
            logits = self._predict_logits(self.test_seen_feature)
            pred = logits.argmax(dim=1).cpu()
            labels = data_preparation.map_label(self.test_seen_label, self.seenclasses)
            target = data_preparation.map_label(self.seenclasses, self.seenclasses)
            score, _, _ = self.compute_per_class_acc_gzsl(labels, pred, target)
            return float(score.item())

    def _target_validation_score(self):
        self.model.eval()
        with torch.no_grad():
            logits = self._predict_logits(self.test_unseen_feature)
            if self.opt.zst:
                target_classes = self.unseenclasses - len(self.seenclasses)
                logits = logits[:, target_classes]
                pred = logits.argmax(dim=1).cpu()
                labels = data_preparation.map_label(self.test_unseen_label, target_classes)
                target = data_preparation.map_label(target_classes, target_classes)
            else:
                pred = logits.argmax(dim=1).cpu()
                labels = data_preparation.map_label(self.test_unseen_label, self.seenclasses)
                target = data_preparation.map_label(self.seenclasses, self.seenclasses)
            score, _, _ = self.compute_per_class_acc_gzsl(labels, pred, target)
            return float(score.item())

    def _predict_logits(self, test_X):
        outputs = []
        for start in range(0, test_X.size(0), self.batch_size):
            end = min(test_X.size(0), start + self.batch_size)
            batch = test_X[start:end]
            if self.cuda:
                batch = batch.cuda()
            logits = self.model.predict(Variable(batch))
            outputs.append(logits.detach().cpu())
        return torch.cat(outputs, dim=0)

    def eval_full_space(self, test_X, test_label, target_classes, save_tag, calc_entropy=False):
        logits = self._predict_logits(test_X)
        predicted_label = logits.argmax(dim=1)
        acc, acc_per_class, prediction_matrix = self.compute_per_class_acc_gzsl(
            test_label, predicted_label, target_classes
        )
        if self.opt.save_pred_matrix:
            self._save_eval_artifacts(acc_per_class, prediction_matrix, len(test_X), len(target_classes), save_tag)
        if calc_entropy:
            from torch.distributions import Categorical

            sm = torch.nn.Softmax(dim=1)
            mean_entropy = Categorical(probs=sm(logits)).entropy().mean()
            print("Mean entropy (log e) of output distributions over test samples: ", mean_entropy)
        return acc

    def eval_with_class_subset(self, test_X, test_label, target_classes, class_indices, save_tag, calc_entropy=False):
        logits = self._predict_logits(test_X)
        subset_logits = logits[:, class_indices]
        predicted_label = subset_logits.argmax(dim=1)
        acc, acc_per_class, prediction_matrix = self.compute_per_class_acc_gzsl(
            test_label, predicted_label, target_classes
        )
        if self.opt.save_pred_matrix:
            self._save_eval_artifacts(acc_per_class, prediction_matrix, len(test_X), len(target_classes), save_tag)
        if calc_entropy:
            from torch.distributions import Categorical

            sm = torch.nn.Softmax(dim=1)
            mean_entropy = Categorical(probs=sm(subset_logits)).entropy().mean()
            print("Mean entropy (log e) of output distributions over test samples: ", mean_entropy)
        return acc

    def _save_eval_artifacts(self, acc_per_class, prediction_matrix, test_len, target_len, save_tag):
        output_dir = os.path.join(self.opt.rootpath, "outputs")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        acc_pt = os.path.join(
            output_dir,
            f"percls_acc_{self.opt.dataset}_{save_tag}_len_test_{test_len}_len_tar_{target_len}.pt",
        )
        acc_txt = os.path.join(
            output_dir,
            f"percls_acc_{self.opt.dataset}_{save_tag}_len_test_{test_len}_len_tar_{target_len}_{self.seedinfo}.txt",
        )
        pred_pt = os.path.join(
            output_dir,
            f"pred_matrix_{self.opt.dataset}_{save_tag}_len_test_{test_len}_len_tar_{target_len}.pt",
        )
        pred_txt = os.path.join(
            output_dir,
            f"pred_matrix_{self.opt.dataset}_{save_tag}_len_test_{test_len}_len_tar_{target_len}_{self.seedinfo}.txt",
        )

        torch.save(acc_per_class, acc_pt)
        np.savetxt(acc_txt, acc_per_class.detach().cpu().numpy(), fmt="%.6f")
        torch.save(prediction_matrix, pred_pt)
        np.savetxt(pred_txt, prediction_matrix.detach().cpu().numpy(), fmt="%.6f")
