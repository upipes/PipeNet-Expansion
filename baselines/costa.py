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

class COSTA(REGRESSOR):
    def __init__(self, opt, **kwargs):
        super().__init__(opt=opt, **kwargs)
        self.opt = opt
        self.unseen_model = model.LINEAR(self.input_dim, len(self.unseenclasses))
        self.ext_model = model.LINEAR(self.input_dim, self.nclass)
        if self.cuda:
            self.unseen_model.cuda()
            self.ext_model.cuda()
        
        self.ext_model.fc.weight.data[:len(self.seenclasses), :] = self.target_weights[:, :2048]
        self.ext_model.fc.bias.data[:len(self.seenclasses)] = self.target_weights[:, 2048]
        for n, unseen_att in enumerate(self.attribute[self.unseenclasses]):
            cooccs = unseen_att.unsqueeze(0) * self.attribute[self.seenclasses]
            norm_coocs = torch.sum(cooccs, dim=-1) / (cooccs.sum() + 10e-5)
            if self.opt.cuda:
                norm_coocs = norm_coocs.cuda()
            pred_weights = torch.sum(norm_coocs[:, None]*self.target_weights, dim=0)
            
            self.unseen_model.fc.weight.data[n, :] = pred_weights[:-1]
            self.unseen_model.fc.bias.data[n] = pred_weights[-1]

            self.ext_model.fc.weight.data[len(self.seenclasses) + n, :] = pred_weights[:-1] 
            self.ext_model.fc.bias.data[len(self.seenclasses) + n] = pred_weights[-1]

        else:    
            self.acc_gzsl, self.acc_seen, self.acc_unseen, self.H, self.acc_unseen_zsl = self.val_gzsl()
