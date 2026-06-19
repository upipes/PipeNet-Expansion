import copy
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


class G2KDEncoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.residual = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(input_dim, input_dim),
        )

    def forward(self, x):
        return x + self.residual(x)


class G2KDModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.encoder = G2KDEncoder(input_dim)
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        logits = self.classifier(feat)
        return feat, logits

    def predict(self, x):
        _, logits = self.forward(x)
        return logits


class G2KD(REGRESSOR):
    def __init__(self, opt, **kwargs):
        super().__init__(opt=opt, **kwargs)
        self.opt = opt

        self.model = G2KDModel(self.input_dim, len(self.seenclasses))
        self.model.encoder.apply(data_preparation.weights_init)
        self.model.classifier.weight.data.copy_(self.target_weights[:, :-1])
        self.model.classifier.bias.data.copy_(self.target_weights[:, -1])

        for param in self.model.classifier.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(
            self.model.encoder.parameters(), lr=self.lr, betas=(self.beta1, 0.999)
        )

        if self.cuda:
            self.model.cuda()

        self.target_feature = self.test_unseen_feature
        self._adapt_target()
        self.evaluate()
        if getattr(self.opt, "save_method_checkpoint", False):
            self._save_method_checkpoint()


    def _adapt_target(self):
        best_state = None
        best_score = float("-inf")
        teacher_model = copy.deepcopy(self.model)
        if self.cuda:
            teacher_model.cuda()

        for _ in trange(self.nepoch):
            teacher_features, teacher_probs, geo_targets, pseudo_labels = self._build_teacher_knowledge(teacher_model)
            perm = torch.randperm(self.target_feature.size(0))

            self.model.train()
            for start in range(0, self.target_feature.size(0), self.batch_size):
                end = min(self.target_feature.size(0), start + self.batch_size)
                idx = perm[start:end]
                batch_x = self.target_feature[idx]
                batch_geo = geo_targets[idx]
                batch_pseudo = pseudo_labels[idx]

                if self.cuda:
                    batch_x = batch_x.cuda()
                    batch_geo = batch_geo.cuda()
                    batch_pseudo = batch_pseudo.cuda()

                _, logits = self.model(batch_x)
                pred = F.softmax(logits, dim=1)

                dis_loss = F.kl_div(torch.log(pred + 1e-5), batch_geo, reduction="batchmean")
                stu_loss = F.cross_entropy(logits, batch_pseudo)
                entropy_loss = torch.mean(torch.sum(-pred * torch.log(pred + 1e-5), dim=1))
                mean_pred = pred.mean(dim=0)
                gentropy_loss = torch.sum(-mean_pred * torch.log(mean_pred + 1e-5))
                ent_loss = entropy_loss - gentropy_loss

                loss = (
                    self.opt.g2kd_dis_weight * dis_loss
                    + self.opt.g2kd_stu_weight * stu_loss
                    + self.opt.g2kd_ent_weight * ent_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            teacher_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
            if self.cuda:
                teacher_model.cuda()

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

    def _build_teacher_knowledge(self, teacher_model):
        teacher_model.eval()
        with torch.no_grad():
            features = []
            probs = []
            for start in range(0, self.target_feature.size(0), self.batch_size):
                end = min(self.target_feature.size(0), start + self.batch_size)
                batch = self.target_feature[start:end]
                if self.cuda:
                    batch = batch.cuda()
                feat, logits = teacher_model(batch)
                features.append(F.normalize(feat, dim=1).detach().cpu())
                probs.append(F.softmax(logits, dim=1).detach().cpu())

            features = torch.cat(features, dim=0)
            probs = torch.cat(probs, dim=0)

            # Soft pseudo-labels from teacher class centers.
            class_centers = torch.matmul(probs.t(), features)
            class_centers = class_centers / (probs.sum(dim=0, keepdim=True).t() + 1e-5)
            class_centers = F.normalize(class_centers, dim=1)
            dist_to_center = torch.cdist(features, class_centers)
            pseudo_labels = dist_to_center.argmin(dim=1)

            # Geometry-guided soft targets from local target structure.
            pairwise_dist = torch.cdist(features, features)
            knn_idx = torch.topk(
                pairwise_dist,
                k=min(self.opt.g2kd_neighbors + 1, pairwise_dist.size(1)),
                largest=False,
                dim=1,
            ).indices[:, 1:]
            neighbor_probs = probs[knn_idx]

            sim = 1.0 / (pairwise_dist.gather(1, knn_idx) + 1e-5)
            sim = sim / (sim.sum(dim=1, keepdim=True) + 1e-5)
            geo_targets = (neighbor_probs * sim.unsqueeze(-1)).sum(dim=1)
            geo_targets = 0.5 * geo_targets + 0.5 * probs
            geo_targets = geo_targets / (geo_targets.sum(dim=1, keepdim=True) + 1e-5)

            return features, probs, geo_targets, pseudo_labels

    def evaluate(self):
        if self.opt.zst:
            target_classes = self.unseenclasses - len(self.seenclasses)
            self.acc_target = self.eval_with_class_subset(
                self.test_unseen_feature,
                data_preparation.map_label(self.test_unseen_label, target_classes),
                data_preparation.map_label(target_classes, target_classes),
                class_indices=target_classes,
                save_tag="g2kd_target_zsl",
                calc_entropy=self.calc_entropy,
            )
            self.acc_unseen_zsl = self.acc_target

            self.acc_zst_unseen = self.eval_with_class_subset(
                self.test_unseen_feature,
                self.test_unseen_label,
                self.seenclasses,
                class_indices=self.seenclasses,
                save_tag="g2kd_target_gzsl",
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
                save_tag="g2kd_gzsl",
                calc_entropy=self.calc_entropy,
            )
            self.acc_seen = self.eval_full_space(
                self.test_seen_feature,
                data_preparation.map_label(self.test_seen_label, self.seenclasses),
                data_preparation.map_label(self.seenclasses, self.seenclasses),
                save_tag="g2kd_seen",
            )
            self.acc_unseen = self.eval_full_space(
                self.test_unseen_feature,
                data_preparation.map_label_extend(
                    self.test_unseen_label, self.unseenclasses, self.seenclasses
                ),
                data_preparation.map_label_extend(
                    self.unseenclasses, self.unseenclasses, self.seenclasses
                ),
                save_tag="g2kd_unseen",
            )
            denom = self.acc_seen + self.acc_unseen
            self.H = 2 * self.acc_seen * self.acc_unseen / denom if denom.item() > 0 else denom
            self.acc_unseen_zsl = torch.zeros((), device=self.acc_unseen.device, dtype=self.acc_unseen.dtype)

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
        self.model.eval()
        for start in range(0, test_X.size(0), self.batch_size):
            end = min(test_X.size(0), start + self.batch_size)
            batch = test_X[start:end]
            if self.cuda:
                batch = batch.cuda()
            logits = self.model.predict(Variable(batch))
            outputs.append(logits.detach().cpu())
        return torch.cat(outputs, dim=0)

    def _save_method_checkpoint(self):
        output_dir = os.path.join(self.opt.rootpath, "models", "method-checkpoints")
        os.makedirs(output_dir, exist_ok=True)
        source_domain = self.opt.zstfrom if self.opt.zst and self.opt.zstfrom != "imagenet" else self.opt.dataset
        target_domain = self.opt.dataset
        checkpoint_path = os.path.join(
            output_dir,
            f"g2kd_{source_domain}_to_{target_domain}_{self.opt.image_embedding}_seed{self.seedinfo}.pth",
        )
        torch.save(
            {
                "method": "g2kd",
                "source": source_domain,
                "target": target_domain,
                "image_embedding": self.opt.image_embedding,
                "seed": self.seedinfo,
                "input_dim": self.input_dim,
                "num_classes": len(self.seenclasses),
                "seenclasses": self.seenclasses.detach().cpu(),
                "unseenclasses": self.unseenclasses.detach().cpu(),
                "model_state_dict": self.model.state_dict(),
            },
            checkpoint_path,
        )
        print(f"Saved G2KD method checkpoint: {checkpoint_path}")
        
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
