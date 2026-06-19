import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import utility.data_preparation as data_preparation
import copy
import random
import sys
import os
import math
import torchvision
from torch.utils.data import Dataset, DataLoader
import utility.model_bases as model
from regressor import REGRESSOR

class ConSE(REGRESSOR):
    def __init__(self, opt, **kwargs):
        super().__init__(opt=opt, **kwargs)
        self.opt = opt
        if self.cuda:
            self.model.cuda()

        # GZSL
        self.acc_gzsl = self.conse_val(self.model, torch.cat((self.test_seen_feature, self.test_unseen_feature), 0),
                        torch.cat((data_preparation.map_label(self.test_seen_label, self.seenclasses), data_preparation.map_label_extend(self.test_unseen_label, self.unseenclasses, self.seenclasses)), 0),
                        torch.cat((data_preparation.map_label(self.seenclasses, self.seenclasses) , data_preparation.map_label_extend(self.unseenclasses, self.unseenclasses, self.seenclasses)), 0),
                        train_attributes=self.attribute[self.seenclasses], test_attributes=torch.cat((self.attribute[self.seenclasses], self.attribute[self.unseenclasses])))    
        self.acc_seen = self.conse_val(self.model, self.test_seen_feature, data_preparation.map_label(self.test_seen_label, self.seenclasses), data_preparation.map_label(self.seenclasses, self.seenclasses), train_attributes=self.attribute[self.seenclasses], test_attributes=torch.cat((self.attribute[self.seenclasses], self.attribute[self.unseenclasses])))    
        self.acc_unseen = self.conse_val(self.model, self.test_unseen_feature, data_preparation.map_label(self.test_unseen_label, self.unseenclasses), data_preparation.map_label(self.unseenclasses, self.unseenclasses), train_attributes=self.attribute[self.seenclasses], test_attributes=torch.cat((self.attribute[self.seenclasses], self.attribute[self.unseenclasses])))
        self.H = 2*self.acc_seen*self.acc_unseen / (self.acc_seen+self.acc_unseen)
        # ZSL 
        self.acc_unseen_zsl = self.conse_val(self.model, self.test_unseen_feature, data_preparation.map_label(self.test_unseen_label, self.unseenclasses), data_preparation.map_label(self.unseenclasses, self.unseenclasses), train_attributes=self.attribute[self.seenclasses], test_attributes=self.attribute[self.unseenclasses])    
            
    def conse_val(self, model, test_X, test_label, target_classes, train_attributes, test_attributes):
        """ Predict semantic embedding for input, then compare to class embeddings (attributes) """
        cos = nn.CosineSimilarity(dim=1, eps=1e-8)
        soft = torch.nn.Softmax(dim=1)
        if self.cuda:
            train_attributes = train_attributes.cuda()
            test_attributes = test_attributes.cuda()
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                logits = model(Variable(test_X[start:end].cuda()))
            else:
                logits = model(Variable(test_X[start:end]))

            if self.opt.class_reduction_ablation:
                probs = soft(logits[:, self.perm])
                pred_embeds = torch.sum(train_attributes[self.perm] * probs.unsqueeze(-1), dim=1)
            else:
                probs = soft(logits)
                pred_embeds = torch.sum(train_attributes * probs.unsqueeze(-1), dim=1)

            output = []
            for pred_embed in pred_embeds: 
                sims = cos(pred_embed[None, :], test_attributes)
                _, idx = torch.max(sims, dim=0)
                output.append(idx)
            
            output = torch.stack(output)
            predicted_label[start:end] = output
            start = end
        
        acc, acc_per_class, prediction_matrix = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        if self.opt.save_pred_matrix:
            torch.save(acc_per_class, self.opt.rootpath + '/outputs/' + self.opt.dataset + 'res101_finetuned_len_test_' + str(len(test_X)) + '_len_tar_' + str(len(target_classes)) + '.pt')
            torch.save(prediction_matrix, self.opt.rootpath + '/outputs/' + self.opt.dataset + 'res101_finetuned_len_test_' + str(len(test_X)) + '_len_tar_' + str(len(target_classes)) + '.pt')

        return acc
