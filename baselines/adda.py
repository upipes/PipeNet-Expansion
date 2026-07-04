import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from tqdm import trange

import utility.data_preparation as data_preparation
from regressor import REGRESSOR


class ADDAEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )

    def forward(self, x):
        return self.net(x)


class ADDAClassifier(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class ADDADiscriminator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)


class ADDA(REGRESSOR):
    def __init__(self, opt, **kwargs):
        super().__init__(train_base=True, opt=opt, **kwargs)
        self.opt = opt

        self.source_encoder = ADDAEncoder(self.input_dim, self.opt.adda_hidden_dim)
        self.target_encoder = ADDAEncoder(self.input_dim, self.opt.adda_hidden_dim)
        self.classifier = ADDAClassifier(self.opt.adda_hidden_dim, len(self.seenclasses))
        self.discriminator = ADDADiscriminator(self.opt.adda_hidden_dim)

        self.source_encoder.apply(data_preparation.weights_init)
        self.target_encoder.apply(data_preparation.weights_init)
        self.classifier.apply(data_preparation.weights_init)
        self.discriminator.apply(data_preparation.weights_init)

        self.class_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()

        self.pretrain_optimizer = optim.Adam(
            list(self.source_encoder.parameters()) + list(self.classifier.parameters()),
            lr=self.lr,
            betas=(self.beta1, 0.999),
        )
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999)
        )
        self.target_optimizer = optim.Adam(
            self.target_encoder.parameters(), lr=self.lr, betas=(self.beta1, 0.999)
        )

        if self.cuda:
            self.source_encoder.cuda()
            self.target_encoder.cuda()
            self.classifier.cuda()
            self.discriminator.cuda()
            self.class_criterion.cuda()
            self.domain_criterion.cuda()

        self.target_feature = self.test_unseen_feature
        self._pretrain_source()
        self._adapt_target()
        self.evaluate()

    def _pretrain_source(self):
        best_state = None
        best_score = float("-inf")
        source_size = self.train_X.size(0)

        for _ in trange(self.opt.adda_pretrain_epochs):
            self.source_encoder.train()
            self.classifier.train()
            perm = torch.randperm(source_size)

            for start in range(0, source_size, self.batch_size):
                end = min(source_size, start + self.batch_size)
                idx = perm[start:end]
                source_x = self.train_X[idx]
                source_y = self.train_Y[idx]

                if self.cuda:
                    source_x = source_x.cuda()
                    source_y = source_y.cuda()

                self.pretrain_optimizer.zero_grad()
                feat = self.source_encoder(source_x)
                logits = self.classifier(feat)
                loss = self.class_criterion(logits, source_y)
                loss.backward()
                self.pretrain_optimizer.step()

            score = self._source_validation_score()
            if score > best_score:
                best_score = score
                best_state = {
                    "source_encoder": {
                        key: value.detach().cpu().clone()
                        for key, value in self.source_encoder.state_dict().items()
                    },
                    "classifier": {
                        key: value.detach().cpu().clone()
                        for key, value in self.classifier.state_dict().items()
                    },
                }

        if best_state is not None:
            self.source_encoder.load_state_dict(best_state["source_encoder"])
            self.classifier.load_state_dict(best_state["classifier"])

        self.target_encoder.load_state_dict(self.source_encoder.state_dict())
        if self.cuda:
            self.source_encoder.cuda()
            self.target_encoder.cuda()
            self.classifier.cuda()

    def _adapt_target(self):
        best_state = None
        best_score = float("-inf")
        source_size = self.train_X.size(0)
        target_size = self.target_feature.size(0)

        self.source_encoder.eval()
        for param in self.source_encoder.parameters():
            param.requires_grad = False

        for _ in trange(self.nepoch):
            self.target_encoder.train()
            self.discriminator.train()
            source_perm = torch.randperm(source_size)
            target_perm = torch.randperm(target_size)
            num_batches = max(
                1,
                max(
                    (source_size + self.batch_size - 1) // self.batch_size,
                    (target_size + self.batch_size - 1) // self.batch_size,
                ),
            )

            for batch_idx in range(num_batches):
                src_start = (batch_idx * self.batch_size) % source_size
                tgt_start = (batch_idx * self.batch_size) % target_size
                src_end = min(src_start + self.batch_size, source_size)
                tgt_end = min(tgt_start + self.batch_size, target_size)
                src_idx = source_perm[src_start:src_end]
                tgt_idx = target_perm[tgt_start:tgt_end]
                if src_idx.numel() == 0 or tgt_idx.numel() == 0:
                    continue

                source_x = self.train_X[src_idx]
                target_x = self.target_feature[tgt_idx]
                if self.cuda:
                    source_x = source_x.cuda()
                    target_x = target_x.cuda()

                with torch.no_grad():
                    source_feat = self.source_encoder(source_x)
                target_feat = self.target_encoder(target_x)

                self.discriminator_optimizer.zero_grad()
                source_domain_logits = self.discriminator(source_feat.detach())
                target_domain_logits = self.discriminator(target_feat.detach())
                source_domain_labels = torch.zeros(source_feat.size(0), dtype=torch.long, device=source_feat.device)
                target_domain_labels = torch.ones(target_feat.size(0), dtype=torch.long, device=target_feat.device)
                d_loss = self.domain_criterion(source_domain_logits, source_domain_labels)
                d_loss = d_loss + self.domain_criterion(target_domain_logits, target_domain_labels)
                d_loss.backward()
                self.discriminator_optimizer.step()

                self.target_optimizer.zero_grad()
                target_feat = self.target_encoder(target_x)
                fool_logits = self.discriminator(target_feat)
                fool_labels = torch.zeros(target_feat.size(0), dtype=torch.long, device=target_feat.device)
                adv_loss = self.domain_criterion(fool_logits, fool_labels)
                (self.opt.adda_adv_weight * adv_loss).backward()
                self.target_optimizer.step()

            score = self._target_validation_score()
            if score > best_score:
                best_score = score
                best_state = {
                    "target_encoder": {
                        key: value.detach().cpu().clone()
                        for key, value in self.target_encoder.state_dict().items()
                    },
                    "discriminator": {
                        key: value.detach().cpu().clone()
                        for key, value in self.discriminator.state_dict().items()
                    },
                }

        if best_state is not None:
            self.target_encoder.load_state_dict(best_state["target_encoder"])
            self.discriminator.load_state_dict(best_state["discriminator"])
            if self.cuda:
                self.target_encoder.cuda()
                self.discriminator.cuda()

    def evaluate(self):
        if self.opt.zst:
            target_classes = self.unseenclasses - len(self.seenclasses)
            self.acc_target = self.eval_with_class_subset(
                self.test_unseen_feature,
                data_preparation.map_label(self.test_unseen_label, target_classes),
                data_preparation.map_label(target_classes, target_classes),
                class_indices=target_classes,
                save_tag="adda_target_zsl",
                calc_entropy=self.calc_entropy,
            )
            self.acc_unseen_zsl = self.acc_target

            self.acc_zst_unseen = self.eval_with_class_subset(
                self.test_unseen_feature,
                self.test_unseen_label,
                self.seenclasses,
                class_indices=self.seenclasses,
                save_tag="adda_target_gzsl",
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
                save_tag="adda_gzsl",
                calc_entropy=self.calc_entropy,
            )
            self.acc_seen = self.eval_full_space(
                self.test_seen_feature,
                data_preparation.map_label(self.test_seen_label, self.seenclasses),
                data_preparation.map_label(self.seenclasses, self.seenclasses),
                save_tag="adda_seen",
            )
            self.acc_unseen = self.eval_full_space(
                self.test_unseen_feature,
                data_preparation.map_label_extend(
                    self.test_unseen_label, self.unseenclasses, self.seenclasses
                ),
                data_preparation.map_label_extend(
                    self.unseenclasses, self.unseenclasses, self.seenclasses
                ),
                save_tag="adda_unseen",
            )
            denom = self.acc_seen + self.acc_unseen
            self.H = 2 * self.acc_seen * self.acc_unseen / denom if denom.item() > 0 else denom
            self.acc_unseen_zsl = torch.zeros((), device=self.acc_unseen.device, dtype=self.acc_unseen.dtype)

    def _source_validation_score(self):
        self.source_encoder.eval()
        self.classifier.eval()
        with torch.no_grad():
            logits = self._predict_logits(self.test_seen_feature, use_target_encoder=False)
            pred = logits.argmax(dim=1).cpu()
            labels = data_preparation.map_label(self.test_seen_label, self.seenclasses)
            target = data_preparation.map_label(self.seenclasses, self.seenclasses)
            score, _, _ = self.compute_per_class_acc_gzsl(labels, pred, target)
            return float(score.item())

    def _target_validation_score(self):
        self.target_encoder.eval()
        self.classifier.eval()
        with torch.no_grad():
            logits = self._predict_logits(self.test_unseen_feature, use_target_encoder=True)
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

    def _predict_logits(self, test_X, use_target_encoder=True):
        encoder = self.target_encoder if use_target_encoder else self.source_encoder
        encoder.eval()
        self.classifier.eval()
        outputs = []
        for start in range(0, test_X.size(0), self.batch_size):
            end = min(test_X.size(0), start + self.batch_size)
            batch = test_X[start:end]
            if self.cuda:
                batch = batch.cuda()
            feat = encoder(Variable(batch))
            logits = self.classifier(feat)
            outputs.append(logits.detach().cpu())
        return torch.cat(outputs, dim=0)

    def eval_full_space(self, test_X, test_label, target_classes, save_tag, calc_entropy=False):
        logits = self._predict_logits(test_X, use_target_encoder=True)
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
        logits = self._predict_logits(test_X, use_target_encoder=True)
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
