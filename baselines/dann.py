import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Function, Variable
from tqdm import trange

import utility.data_preparation as data_preparation
from regressor import REGRESSOR


class GradientReverse(Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


class DANNNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
        )
        self.label_classifier = nn.Linear(hidden_dim, num_classes)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, lambd=1.0):
        feat = self.feature_extractor(x)
        class_logits = self.label_classifier(feat)
        rev_feat = GradientReverse.apply(feat, lambd)
        domain_logits = self.domain_classifier(rev_feat)
        return class_logits, domain_logits

    def predict(self, x):
        feat = self.feature_extractor(x)
        return self.label_classifier(feat)


class DANN(REGRESSOR):
    def __init__(self, opt, **kwargs):
        super().__init__(train_base=True, opt=opt, **kwargs)
        self.opt = opt

        self.model = DANNNet(
            input_dim=self.input_dim,
            hidden_dim=self.opt.dann_hidden_dim,
            num_classes=len(self.seenclasses),
        )
        self.model.apply(data_preparation.weights_init)
        self.class_criterion = nn.CrossEntropyLoss()
        self.domain_criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.lr, betas=(self.beta1, 0.999)
        )

        if self.cuda:
            self.model.cuda()
            self.class_criterion.cuda()
            self.domain_criterion.cuda()

        self.target_feature = self.test_unseen_feature
        self.fit()
        self.evaluate()

    def fit(self):
        best_model = None
        best_score = float("-inf")
        source_size = self.train_X.size(0)
        target_size = self.target_feature.size(0)

        for epoch in trange(self.nepoch):
            self.model.train()
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
                src_idx = source_perm[
                    (batch_idx * self.batch_size) % source_size : ((batch_idx + 1) * self.batch_size) % source_size
                ]
                if src_idx.numel() == 0:
                    start = (batch_idx * self.batch_size) % source_size
                    end = min(start + self.batch_size, source_size)
                    src_idx = source_perm[start:end]

                tgt_idx = target_perm[
                    (batch_idx * self.batch_size) % target_size : ((batch_idx + 1) * self.batch_size) % target_size
                ]
                if tgt_idx.numel() == 0:
                    start = (batch_idx * self.batch_size) % target_size
                    end = min(start + self.batch_size, target_size)
                    tgt_idx = target_perm[start:end]

                source_x = self.train_X[src_idx]
                source_y = self.train_Y[src_idx]
                target_x = self.target_feature[tgt_idx]

                if self.cuda:
                    source_x = source_x.cuda()
                    source_y = source_y.cuda()
                    target_x = target_x.cuda()

                p = float(epoch * num_batches + batch_idx) / float(max(1, self.nepoch * num_batches))
                lambd = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

                self.optimizer.zero_grad()

                class_logits, source_domain_logits = self.model(source_x, lambd=lambd)
                _, target_domain_logits = self.model(target_x, lambd=lambd)

                class_loss = self.class_criterion(class_logits, source_y)
                source_domain_labels = torch.zeros(source_x.size(0), dtype=torch.long, device=source_x.device)
                target_domain_labels = torch.ones(target_x.size(0), dtype=torch.long, device=target_x.device)
                domain_loss = self.domain_criterion(source_domain_logits, source_domain_labels)
                domain_loss = domain_loss + self.domain_criterion(target_domain_logits, target_domain_labels)

                loss = class_loss + self.opt.dann_domain_weight * domain_loss
                loss.backward()
                self.optimizer.step()

            score = self._validation_score()
            if score > best_score:
                best_score = score
                best_model = {
                    key: value.detach().cpu().clone()
                    for key, value in self.model.state_dict().items()
                }

        if best_model is not None:
            self.model.load_state_dict(best_model)
            if self.cuda:
                self.model.cuda()

    def evaluate(self):
        if self.opt.zst:
            target_classes = self.unseenclasses - len(self.seenclasses)
            self.acc_target = self.eval_with_class_subset(
                self.test_unseen_feature,
                data_preparation.map_label(self.test_unseen_label, target_classes),
                data_preparation.map_label(target_classes, target_classes),
                class_indices=target_classes,
                save_tag="dann_target_zsl",
                calc_entropy=self.calc_entropy,
            )
            self.acc_unseen_zsl = self.acc_target

            self.acc_zst_unseen = self.eval_with_class_subset(
                self.test_unseen_feature,
                self.test_unseen_label,
                self.seenclasses,
                class_indices=self.seenclasses,
                save_tag="dann_target_gzsl",
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
                save_tag="dann_gzsl",
                calc_entropy=self.calc_entropy,
            )
            self.acc_seen = self.eval_full_space(
                self.test_seen_feature,
                data_preparation.map_label(self.test_seen_label, self.seenclasses),
                data_preparation.map_label(self.seenclasses, self.seenclasses),
                save_tag="dann_seen",
            )
            self.acc_unseen = self.eval_full_space(
                self.test_unseen_feature,
                data_preparation.map_label_extend(self.test_unseen_label, self.unseenclasses, self.seenclasses),
                data_preparation.map_label_extend(self.unseenclasses, self.unseenclasses, self.seenclasses),
                save_tag="dann_unseen",
            )
            denom = self.acc_seen + self.acc_unseen
            self.H = 2 * self.acc_seen * self.acc_unseen / denom if denom.item() > 0 else denom
            self.acc_unseen_zsl = torch.zeros((), device=self.acc_unseen.device, dtype=self.acc_unseen.dtype)

    def _validation_score(self):
        self.model.eval()
        with torch.no_grad():
            logits = self._predict_logits(self.test_unseen_feature)
            if self.opt.zst:
                target_classes = self.unseenclasses - len(self.seenclasses)
                subset_logits = logits[:, target_classes]
                pred = subset_logits.argmax(dim=1).cpu()
                labels = data_preparation.map_label(self.test_unseen_label, target_classes)
                target = data_preparation.map_label(target_classes, target_classes)
            else:
                pred = logits.argmax(dim=1).cpu()
                labels = data_preparation.map_label(self.test_seen_label, self.seenclasses)
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

    def eval_with_class_subset(
        self,
        test_X,
        test_label,
        target_classes,
        class_indices,
        save_tag,
        calc_entropy=False,
    ):
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
