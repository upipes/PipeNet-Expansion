import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import utility.data_preparation as data_preparation
import copy
import utility.model_bases as model
from regressor import REGRESSOR
from tqdm import trange

class BASECLASSIFIER(REGRESSOR):
    def __init__(self, _train_X, _train_Y, data_loader, _nclass, _cuda, seedinfo, train_base=True, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, _embed_dim=1000, _num_layers=3, opt=None):
        super().__init__(_train_X, _train_Y, data_loader, _nclass, _cuda, seedinfo, train_base, _lr, _beta1, _nepoch, _batch_size, _embed_dim, _num_layers, opt)
        self.opt = opt
        self.train_base = train_base
        self.nepoch = _nepoch
        self.model = model.LINEAR(self.input_dim, len(self.seenclasses))
        self.model.apply(data_preparation.weights_init)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer_classifier = optim.Adam(self.model.parameters(), lr=self.opt.classifier_lr, betas=(self.opt.classifier_beta1, 0.999))
        self.input = torch.FloatTensor(_batch_size, self.input_dim)
        self.label = torch.LongTensor(_batch_size)

        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.ntrain = self.train_X.size()[0]

    def fit(self):
        best_H = 0
        best_seen = 0
        best_unseen = 0

        if self.train_base:
            for epoch in trange(self.nepoch):
                for i in range(0, self.ntrain, self.batch_size):
                    self.model.zero_grad()
                    batch_input, batch_label = self.next_batch(self.batch_size)
                    self.input.copy_(batch_input)
                    self.label.copy_(batch_label)

                    inputv = Variable(self.input)
                    labelv = Variable(self.label)
                    output = self.model(inputv)
                    loss = self.criterion(output, labelv)
                    loss.backward()
                    self.optimizer_classifier.step()

                acc_val_seen = 0
                acc_train = self.val_model(self.model, self.train_X, self.train_Y, data_preparation.map_label(self.seenclasses, self.seenclasses))
                acc_val_seen = self.val_model(self.model, self.test_seen_feature, data_preparation.map_label(self.test_seen_label, self.seenclasses), data_preparation.map_label(self.seenclasses, self.seenclasses))
                if acc_val_seen > best_seen:
                    print(f'New best validation seen class accuracy={acc_val_seen*100:.2f}% (train seen class accuracy={acc_train*100:.2f}%)')
                    best_seen = acc_val_seen
                    best_model = copy.deepcopy(self.model)
        else:
            if self.opt.zst:
                best_model = torch.load(self.opt.rootpath + '/models/base-classifiers/' + self.opt.zstfrom + '_' + self.opt.image_embedding + f'_seed{self.seedinfo}_clr{self.opt.classifier_lr}_nep{self.nepoch}')
            else:
                best_model = torch.load(self.opt.rootpath + '/models/base-classifiers/' + self.opt.dataset + '_' + self.opt.image_embedding + f'_seed{self.seedinfo}_clr{self.opt.classifier_lr}_nep{self.nepoch}')
        return best_model

    def next_batch(self, batch_size):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
        # the last batch
        if start + batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.train_X[start:self.ntrain]
                Y_rest_part = self.train_Y[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.train_X = self.train_X[perm]
            self.train_Y = self.train_Y[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.train_X[start:end]
            Y_new_part = self.train_Y[start:end]
            if rest_num_examples > 0:
                return torch.cat((X_rest_part, X_new_part), 0), torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                return X_new_part, Y_new_part
        else:
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            # from index start to index end-1
            return self.train_X[start:end], self.train_Y[start:end]
