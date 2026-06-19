import os

import numpy as np
import torch
from torch.autograd import Variable

import utility.data_preparation as data_preparation
from regressor import REGRESSOR

class SourceOnly(REGRESSOR):
    def __init__(self, opt, **kwargs):
        super().__init__(opt=opt, **kwargs)
        self.opt = opt

        if self.cuda:
            self.model.cuda()

        if self.opt.zst:
            self._evaluate_transfer_setting()
        else:
            self._evaluate_default_setting()

    def _evaluate_transfer_setting(self):
        target_classes = self.unseenclasses - len(self.seenclasses)

        self.acc_target = self.eval_with_class_subset(
            self.test_unseen_feature,
            data_preparation.map_label(self.test_unseen_label, target_classes),
            data_preparation.map_label(target_classes, target_classes),
            class_indices=target_classes,
            save_tag="sourceonly_target_zsl",
            calc_entropy=self.calc_entropy,
        )
        self.acc_unseen_zsl = self.acc_target

        self.acc_zst_unseen = self.eval_with_class_subset(
            self.test_unseen_feature,
            self.test_unseen_label,
            self.seenclasses,
            class_indices=self.seenclasses,
            save_tag="sourceonly_target_gzsl",
        )

        self.acc_gzsl = self.acc_zst_unseen
        self.acc_seen = self.acc_zst_unseen.new_zeros(())
        self.acc_unseen = self.acc_zst_unseen
        self.H = self.acc_zst_unseen.new_zeros(())

    def _evaluate_default_setting(self):
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

        self.acc_gzsl = self.val_model(
            self.model,
            gzsl_features,
            gzsl_labels,
            gzsl_target_classes,
            calc_entropy=self.calc_entropy,
        )
        self.acc_seen = self.val_model(
            self.model,
            self.test_seen_feature,
            data_preparation.map_label(self.test_seen_label, self.seenclasses),
            data_preparation.map_label(self.seenclasses, self.seenclasses),
        )
        self.acc_unseen = self.val_model(
            self.model,
            self.test_unseen_feature,
            data_preparation.map_label_extend(
                self.test_unseen_label, self.unseenclasses, self.seenclasses
            ),
            data_preparation.map_label_extend(
                self.unseenclasses, self.unseenclasses, self.seenclasses
            ),
        )

        denom = self.acc_seen + self.acc_unseen
        if denom.item() > 0:
            self.H = 2 * self.acc_seen * self.acc_unseen / denom
        else:
            self.H = denom

        self.acc_unseen_zsl = torch.zeros(
            (), device=self.acc_unseen.device, dtype=self.acc_unseen.dtype
        )

    def eval_with_class_subset(
        self,
        test_X,
        test_label,
        target_classes,
        class_indices,
        save_tag,
        calc_entropy=False,
    ):
        start = 0
        ntest = test_X.size(0)
        predicted_label = torch.LongTensor(test_label.size())
        all_outputs = None
        if calc_entropy:
            all_outputs = torch.empty(ntest, len(class_indices))

        class_indices_device = class_indices.cuda() if self.cuda else class_indices

        for _ in range(0, ntest, self.batch_size):
            end = min(ntest, start + self.batch_size)
            batch = test_X[start:end]
            if self.cuda:
                batch = batch.cuda()
            output = self.model(Variable(batch))
            subset_output = output[:, class_indices_device]
            if calc_entropy:
                all_outputs[start:end] = subset_output.detach().cpu()
            _, batch_pred = torch.max(subset_output.data, 1)
            predicted_label[start:end] = batch_pred.cpu()
            start = end

        acc, acc_per_class, prediction_matrix = self.compute_per_class_acc_gzsl(
            test_label, predicted_label, target_classes
        )

        if self.opt.save_pred_matrix:
            self._save_eval_artifacts(
                acc_per_class, prediction_matrix, len(test_X), len(target_classes), save_tag
            )

        if calc_entropy:
            from torch.distributions import Categorical

            sm = torch.nn.Softmax(dim=1)
            mean_entropy = Categorical(probs=sm(all_outputs)).entropy().mean()
            print("Mean entropy (log e) of output distributions over test samples: ", mean_entropy)

        return acc

    def _save_eval_artifacts(self, acc_per_class, prediction_matrix, test_len, target_len, save_tag):
        output_dir = os.path.join(self.opt.rootpath, "outputs")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        acc_pt = os.path.join(
            output_dir,
            f"percls_acc_{self.opt.dataset}_{save_tag}_len_test_{test_len}_len_tar_{target_len}_{self.seedinfo}.pt",
        )
        acc_txt = os.path.join(
            output_dir,
            f"percls_acc_{self.opt.dataset}_{save_tag}_len_test_{test_len}_len_tar_{target_len}_{self.seedinfo}.txt",
        )
        pred_pt = os.path.join(
            output_dir,
            f"pred_matrix_{self.opt.dataset}_{save_tag}_len_test_{test_len}_len_tar_{target_len}_{self.seedinfo}.pt",
        )
        pred_txt = os.path.join(
            output_dir,
            f"pred_matrix_{self.opt.dataset}_{save_tag}_len_test_{test_len}_len_tar_{target_len}_{self.seedinfo}.txt",
        )

        # torch.save(acc_per_class, acc_pt)
        np.savetxt(acc_txt, acc_per_class.detach().cpu().numpy(), fmt="%.6f")
        # torch.save(prediction_matrix, pred_pt)
        np.savetxt(pred_txt, prediction_matrix.detach().cpu().numpy(), fmt="%.6f")