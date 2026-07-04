import torch
import numpy as np
import torchvision

from baselines.baseline import Baseline
from utility.model_bases import LINEAR
import utility.data_preparation as data_preparation 

class VGSE_CRM(Baseline):
    """ Baseline inspired by the Class Relation Module (CRM) from 
    VGSE: Visually-Grounded Semantic Embeddings for Zero-Shot Learning
    by Xu et al. Contains implementation of both WAvg and SMO CRM. """
    def __init__(self, opt, **kwargs):
        super().__init__(opt=opt, **kwargs)
        self.opt = opt

        if opt.vgse_baseline == 'wavg':
            pred_weights = self.WAvg(opt.vgse_nbs, opt.vgse_eta)
        elif opt.vgse_baseline == 'smo':
            pred_weights = self.SMO(alpha=opt.vgse_alpha)

        self.unseen_model = LINEAR(self.input_dim, len(self.unseenclasses))
        self.ext_model = LINEAR(self.input_dim, self.nclass)
        if self.cuda:
            self.unseen_model.cuda()
            self.ext_model.cuda()

        self.ext_model.fc.weight.data[:len(self.seenclasses), :] = self.target_weights[:, :2048]
        self.ext_model.fc.bias.data[:len(self.seenclasses)] = self.target_weights[:, 2048]

        self.evaluate_weights(pred_weights)

    def WAvg(self, num_neighbours=5, eta=5):
        """ Implementation of Weighted Average (WAvg) CRM.
        Hyperparameters (num_neighbours and eta) taken from paper. """

        unseen_att_sims = np.zeros((len(self.unseenclasses), len(self.seenclasses)))
        for i in range(len(self.unseenclasses)):
            for j in range(len(self.seenclasses)):
                unseen_att_sims[i, j] =  torch.exp(-eta*torch.dist(self.attribute[self.unseenclasses[i]], self.attribute[self.seenclasses[j]]))
        unseen_att_sims = torch.from_numpy(unseen_att_sims).float()
        
        if self.opt.cuda:
            unseen_att_sims = unseen_att_sims.cuda()
        
        pred_weights = torch.matmul(unseen_att_sims, self.target_weights)
        
        return pred_weights

    def SMO(self, alpha=0, eps=10e-8):
        """ Implementation of Similarity Matrix Optimization (SMO) CRM """
        assert alpha in [-1, 0]
        
        loss_fnc = torch.nn.MSELoss()
        reg = torch.nn.L1Loss()
        sum_constraint = torch.ones(1)[0]

        if alpha == 0:
            lr = 1000 
            domain_fnc = torch.nn.Softmax(dim=0)
        else:
            lr = 10e-6 
            domain_fnc = torch.nn.Tanh()

        all_pred_weights = torch.zeros(len(self.unseenclasses), self.target_weights.size(1))
        if self.cuda:
            all_pred_weights = all_pred_weights.cuda()
            sum_constraint = sum_constraint.cuda()
            self.attribute = self.attribute.cuda()

        for i in range(len(self.unseenclasses)):
            converged = False
            best_loss = 1000
            prev_loss = 1000
            counter = 0

            smo = SMOModel(domain_fnc=domain_fnc, dim=len(self.seenclasses))
            if self.cuda:
                smo.cuda()
            
            optim = torch.optim.SGD(smo.parameters(), lr=lr)
            
            while not converged:
                optim.zero_grad()
                pred_att = smo(self.attribute[self.seenclasses])
                loss = loss_fnc(pred_att, self.attribute[self.unseenclasses[i]]) - alpha * reg(torch.sum(smo.domain_fnc(smo.r)), sum_constraint)
                loss.backward()
                optim.step()
                
                if loss < best_loss:
                    best_loss = loss
                    best_r = smo.r
                    if torch.abs(prev_loss - loss) < eps:
                        counter += 1
                    else:
                        counter = 0
                else:
                    counter += 1

                if counter > 10:
                    converged = True
                
                prev_loss = loss
                
            pred_weights = torch.sum(domain_fnc(best_r)[:, None] * self.target_weights, dim=0) 
            all_pred_weights[i,:] = pred_weights
            
        return all_pred_weights


class SMOModel(torch.nn.Module):
    def __init__(self, domain_fnc, dim):
        super().__init__()
        self.domain_fnc = domain_fnc
        self.dim = dim
        self.r = torch.nn.parameter.Parameter(data=torch.normal(mean=torch.zeros(dim), std=2/dim), requires_grad=True)
        
    def forward(self, data):
        pred_att = torch.sum(self.domain_fnc(self.r)[:, None] * data, dim=0)
        return pred_att
