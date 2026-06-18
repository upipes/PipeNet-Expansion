import torch
import torch.nn as nn
import torch.optim as optim
import utility.data_preparation as data_preparation
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
import copy

from regressor import REGRESSOR
import utility.model_bases as model

from baselines.wDAEGNN.low_shot_learning.architectures.classifiers.weights_denoising_autoencoder import WeightsDAE


class Joint(REGRESSOR):
    def __init__(self, _train_X, _train_Y, data_loader, _nclass, _cuda, seedinfo, train_base=False, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=100, _embed_dim=1000, _num_layers=3, _unseen_rate=0, opt=None):
        super().__init__(_train_X, _train_Y, data_loader, _nclass, _cuda, seedinfo, train_base, _lr, _beta1, _nepoch, _batch_size, _embed_dim, _num_layers, opt)
        self.opt = opt
        self.seedinfo = seedinfo

        self.lr = _lr
        self.beta1 = _beta1
        self.nepoch = _nepoch
        self.batch_size = _batch_size
        self.embed_dim = _embed_dim
        self.num_layers = _num_layers
        self.nclass = _nclass
        self.cuda = _cuda

        self.test_seen_feature = data_loader.test_seen_feature
        self.test_seen_label = data_loader.test_seen_label
        self.test_unseen_feature = data_loader.test_unseen_feature
        self.test_unseen_label = data_loader.test_unseen_label

        self.seenclasses = data_loader.seenclasses
        self.unseenclasses = data_loader.unseenclasses
        
        if self.opt.conclude_inv:
            self.attribute_f = data_loader.attribute_f
            self.attribute_inv = data_loader.attribute_inv
            self.attribute_new = data_loader.attribute_new
        else:
            self.attribute_new = data_loader.attribute_f
        
        self.unseen_rate = _unseen_rate

        if self.opt.cuda:
            self.target_weights = self.target_weights.cuda()

        # baseline SubReg
        if self.opt.subspace_proj:
            base_weights_mat = torch.cat((self.model.fc.weight.data, self.model.fc.bias.data.unsqueeze(1)), 1)
            tr_base = torch.transpose(base_weights_mat, 0, 1)
            self.Q, self.R = torch.linalg.qr(tr_base, mode='reduced')

        # robustness
        if self.opt.class_reduction_ablation:
            perm = torch.randperm(len(self.seenclasses))
            assert self.opt.class_reduction_ablation in range(1, len(self.seenclasses)+1)
            perm = perm[:self.opt.class_reduction_ablation]
            if self.opt.concatenation and self.opt.conclude_inv:
                training_attributes_f = self.attribute_f[self.seenclasses][perm]
                training_attributes_inv = self.attribute_inv[self.seenclasses][perm]

                training_weights_f = self.target_weights[perm]
                placeholder_weights_inv = torch.zeros(len(self.nclass), self.target_weights.size(1))
                training_weights_inv = placeholder_weights_inv[self.seenclasses][perm]
            else:
                training_attributes = self.attribute_new[self.seenclasses][perm]
                training_weights = self.target_weights[perm]

            temp = self.unseenclasses
            if self.unseen_rate:
                perm2 = torch.randperm(len(temp))
                num = len(temp) * self.unseen_rate // 100
                self.perm2 = perm2[:num]
        
        # ablation
        placeholder_weights_f = torch.zeros(len(self.unseenclasses), self.target_weights.size(1))
        placeholder_weights_inv = torch.zeros(self.nclass, self.target_weights.size(1))
        if opt.single_autoencoder_baseline:
            if opt.class_reduction_ablation:
                if self.opt.concatenation and self.opt.conclude_inv:
                    att2weight_dataset = data_preparation.GenericDatasetINV(opt, _input=training_attributes_f, _target=training_weights_f, _input_inv=training_attributes_inv, _target_inv=training_weights_inv, cuda=self.cuda)
                else:
                    att2weight_dataset = data_preparation.GenericDataset(opt, _input=training_attributes, _target=training_weights, cuda=self.cuda)
                self.loader = DataLoader(att2weight_dataset, batch_size=self.batch_size, shuffle=True)
            else:
                if self.opt.concatenation and self.opt.conclude_inv:
                    att2weight_dataset = data_preparation.GenericDatasetINV(opt, _input=self.attribute_f[self.seenclasses], _target=self.target_weights, _input_inv=self.attribute_inv[self.seenclasses], _target_inv=placeholder_weights_inv[self.seenclasses], cuda=self.cuda)
                else:
                    att2weight_dataset = data_preparation.GenericDataset(opt, _input=self.attribute_new[self.seenclasses], _target=self.target_weights, cuda=self.cuda)
                self.loader = DataLoader(att2weight_dataset, batch_size=self.batch_size, shuffle=True)
        else:
            if opt.class_reduction_ablation:
                if self.opt.concatenation and self.opt.conclude_inv:
                    att2weight_dataset = data_preparation.GenericDatasetINV(opt, _input=training_attributes_f, _target=training_weights_f, _input_inv=training_attributes_inv, _target_inv=training_weights_inv, cuda=self.cuda)
                else:
                    att2weight_dataset = data_preparation.GenericDataset(opt, _input=training_attributes, _target=training_weights, cuda=self.cuda)
            else:
                if self.opt.concatenation and self.opt.conclude_inv:
                    combined_seen_dataset = data_preparation.GenericDatasetINV(opt, _input=self.attribute_f[self.seenclasses], _target=self.target_weights, _input_inv=self.attribute_inv[self.seenclasses], _target_inv=placeholder_weights_inv[self.seenclasses], cuda=self.cuda)
                else:
                    combined_seen_dataset = data_preparation.GenericDataset(opt, _input=self.attribute_new[self.seenclasses], _target=self.target_weights, cuda=self.cuda)
            if self.cuda:
                placeholder_weights_f = placeholder_weights_f.cuda()
                placeholder_weights_inv = placeholder_weights_inv.cuda()
            if opt.class_reduction_ablation:
                if self.opt.concatenation and self.opt.conclude_inv:
                    combined_full_dataset = data_preparation.GenericDatasetINV(opt, _input=torch.cat((training_attributes_f, self.attribute_f[self.unseenclasses])), 
                                                                            _target=torch.cat((training_weights_f, placeholder_weights_f), dim=0),
                                                                            _input_inv=torch.cat((training_attributes_inv, self.attribute_inv[self.unseenclasses])), 
                                                                            _target_inv=torch.cat((training_weights_inv, placeholder_weights_inv[self.unseenclasses]), dim=0), cuda=self.cuda)
                else:
                    combined_full_dataset = data_preparation.GenericDataset(opt, _input=torch.cat((training_attributes, self.attribute_new[self.unseenclasses])), 
                                                                            _target=torch.cat((training_weights, placeholder_weights_f), dim=0), cuda=self.cuda)
            else:
                if self.opt.concatenation and self.opt.conclude_inv:
                    combined_full_dataset = data_preparation.GenericDatasetINV(opt, _input=torch.cat((self.attribute_f[self.seenclasses], self.attribute_f[self.unseenclasses])), 
                                                                            _target=torch.cat((self.target_weights, placeholder_weights_f), dim=0),
                                                                            _input_inv=torch.cat((self.attribute_inv[self.seenclasses], self.attribute_inv[self.unseenclasses])), 
                                                                            _target_inv=torch.cat((placeholder_weights_inv[self.seenclasses], placeholder_weights_inv[self.unseenclasses]), dim=0), cuda=self.cuda)
                else:
                    combined_full_dataset = data_preparation.GenericDataset(opt, _input=torch.cat((self.attribute_new[self.seenclasses], self.attribute_new[self.unseenclasses])), 
                                                                            _target=torch.cat((self.target_weights, placeholder_weights_f), dim=0), cuda=self.cuda)
            if self.opt.include_unseen:
                self.loader = DataLoader(combined_full_dataset, batch_size=self.batch_size, shuffle=True)
            else:
                self.loader = DataLoader(combined_seen_dataset, batch_size=self.batch_size, shuffle=True)
        
        if self.opt.factual_branch in ['attention', 'mean']:
            if self.opt.concatenation and self.opt.conclude_inv:
                self.AE_attribute = model.ATT_AUTOENCODER_inv(self.opt, input_dim=self.attribute_f.size(1), att_dim=self.opt.att_dim, embed_dim=self.embed_dim, wordemb_dim=self.opt.wordemb_dim)
            else:
                self.AE_attribute = model.ATT_AUTOENCODER(self.opt, input_dim=self.attribute_new.size(1), att_dim=self.opt.att_dim, embed_dim=self.embed_dim, wordemb_dim=self.opt.wordemb_dim)
        else:
            self.AE_attribute = model.AUTOENCODER(self.opt, input_dim=self.attribute_new.size(1), embed_dim=self.embed_dim, num_layers=self.num_layers)
        self.AE_weight = model.AUTOENCODER(self.opt, input_dim=self.target_weights.size(1), embed_dim=self.embed_dim, num_layers=self.num_layers)

        self.AE_attribute.apply(data_preparation.weights_init)
        self.AE_weight.apply(data_preparation.weights_init)

        if self.opt.single_autoencoder_baseline:
            if self.opt.factual_branch in ['attention', 'mean']:
                if self.opt.concatenation and self.opt.conclude_inv:
                    self.model = model.ATT_AUTOENCODER_inv(self.opt, input_dim=self.attribute_f.size(1), embed_dim=self.embed_dim, att_dim=self.opt.att_dim, output_dim=self.target_weights.size(1), wordemb_dim=self.opt.wordemb_dim)
                else:
                    self.model = model.ATT_AUTOENCODER(self.opt, input_dim=self.attribute_new.size(1), embed_dim=self.embed_dim, att_dim=self.opt.att_dim, output_dim=self.target_weights.size(1), wordemb_dim=self.opt.wordemb_dim)
            else:
                self.model = model.AUTOENCODER(self.opt, input_dim=self.attribute_new.size(1), embed_dim=self.embed_dim, output_dim=self.target_weights.size(1), num_layers=self.num_layers)
        else:
            self.model = model.JOINT_AUTOENCODER(self.opt, autoencoder1=self.AE_attribute, autoencoder2=self.AE_weight)

        if self.model:
            self.model.apply(data_preparation.weights_init)

        if opt.cos_sim_loss:
            self.criterion = data_preparation.cos_sim_loss(reduction='none')
        else:
            self.criterion = nn.MSELoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.l1loss = nn.L1Loss(reduction='none')

        # baseline wDAG
        if opt.daegnn:
            dae_num_features = 2049 
            self.dae_meta_batch_size = 4 
            if self.opt.single_autoencoder_baseline:
                self.dae_loader = DataLoader(att2weight_dataset, batch_size=len(self.seenclasses), shuffle=False)
            else:
                self.dae_loader = DataLoader(combined_seen_dataset, batch_size=len(self.seenclasses), shuffle=False)
            self.dae = WeightsDAE({
                'gaussian_noise': 0.08,
                'comp_reconstruction_loss': True,
                'targets_as_input': False,
                'dae_type': 'RelationNetBasedGNN',
                'num_layers': 2,
                'num_features_input': dae_num_features,
                'num_features_output': 2 * dae_num_features,
                'num_features_hidden': 3 * dae_num_features,
                'update_dropout': 0.7,

                'nun_features_msg': 3 * dae_num_features,
                'aggregation_dropout': 0.7,
                'topK_neighbors': 10,
                'temperature': 5.0,
                'learn_temperature': False,
            })
            self.dae_optimizer = optim.Adam(self.dae.parameters(), lr=_lr, betas=(_beta1, 0.999), weight_decay=0.0)

        if self.cuda:
            self.AE_attribute.cuda()
            self.AE_weight.cuda()
            if self.model:
                self.model.cuda()
            self.criterion.cuda()
            self.mse_loss.cuda()
            self.l1loss.cuda()
            if opt.daegnn:
                self.dae.cuda()

        if self.unseen_rate:
            self.unseenclasses = self.unseenclasses[self.perm2]

        self.unseen_model = model.LINEAR(self.test_seen_feature.size(1), len(self.unseenclasses))
        self.ext_model = model.LINEAR(self.test_seen_feature.size(1), len(self.seenclasses) + len(self.unseenclasses))
        self.ext_model.fc.weight.data[:len(self.seenclasses), :] = self.target_weights[:, :-1]
        self.ext_model.fc.bias.data[:len(self.seenclasses)] = self.target_weights[:, -1]

        if self.cuda:
            self.ext_model.cuda()
            self.unseen_model.cuda()

        self.lr = _lr
        self.beta1 = _beta1
        self.optimizer_attribute_AE = optim.Adam(self.AE_attribute.parameters(), lr=_lr, betas=(_beta1, 0.999))
        self.optimizer_weight_AE = optim.Adam(self.AE_weight.parameters(), lr=_lr, betas=(_beta1, 0.999))
        if self.model:
            self.weight_optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999), weight_decay=0.0)

        self.index_in_epoch = 0
        self.epochs_completed = 0

    def fit(self):
        run_best_acc_gzsl, run_best_acc_seen, run_best_acc_unseen, run_best_H, run_best_unseen_zsl = 0, 0, 0, 0, 0

        counter = 0
        breaking = False
        epoch_losses = []

        for epoch in trange(self.nepoch):
            epoch_loss = 0
            for i_batch, batch in enumerate(self.loader):
                # seen
                mask = torch.where(torch.sum(torch.abs(batch[1]), dim=-1) > 0., 1., 0.)[:, None]
                if self.cuda:
                    mask = mask.cuda()
                mask_sum = torch.clamp(mask.sum(), min=1.)
                inv_mask_sum = torch.clamp((1-mask).sum(), min=1)

                self.model.zero_grad()

                if self.opt.single_autoencoder_baseline:
                    if self.opt.concatenation:
                        att, weights, att_inv, weights_inv = batch
                        output = self.model(att, att_inv)
                        weights = torch.cat([weights, weights_inv], dim=0)
                        loss = (self.criterion(output, weights)*mask).sum(0).mean()/mask_sum
                        
                        weight_output_f = output[:att.shape[0],:]
                        weight_output_inv = output[att.shape[0]:,:]
                        seperate_loss = - self.criterion(weight_output_f, weight_output_inv).mean()
                        if seperate_loss > self.opt.sep_param:
                            loss = loss + seperate_loss
                    else:
                        att, weights = batch
                        output = self.model(att)
                        loss = self.criterion(output, weights).mean()

                        # baseline SubReg
                        if self.opt.subspace_proj:
                            mut = output @ self.Q
                            mutnorm = mut / torch.norm(self.Q.T, dim=1).unsqueeze(0)
                            proj_weights = mutnorm @ self.Q.T
                            proj_weights = proj_weights.squeeze()
                            subspace_proj_loss = 0.001 * torch.norm(output - proj_weights, dim=-1).mean()
                            loss += subspace_proj_loss
                else:
                    output = self.model(batch)
                    if self.opt.conclude_inv and not self.opt.concatenation:
                        att_from_att, att_from_weight, weight_from_weight, weight_from_att, weight_from_att_inv, latent_att, latent_weight = output
                    else:
                        att_from_att, att_from_weight, weight_from_weight, weight_from_att, latent_att, latent_weight = output

                    if self.opt.concatenation:
                        if not self.opt.inv_merge:
                            placeholder = torch.zeros(batch[0].shape[0], batch[0].shape[1]-batch[2].shape[1])
                            placeholder = placeholder.cuda()
                            batch[2] = torch.concatenate([batch[2], placeholder], axis=1)
                            placeholder = torch.zeros(batch[0].shape[0], batch[1].shape[1]-batch[3].shape[1])
                            placeholder = placeholder.cuda()
                            batch[3] = torch.concatenate([batch[3], placeholder], axis=1)
                            batch[0] = torch.cat([batch[0], batch[2]], dim=0)
                            batch[1] = torch.cat([batch[1], batch[3]], dim=0)
                        else:
                            batch[0] = torch.cat([batch[0], batch[2]], dim=0)
                            batch[1] = torch.cat([batch[1], batch[3]], dim=0)
                        mask = torch.where(torch.sum(torch.abs(batch[1]), dim=-1) > 0., 1., 0.)[:, None]
                        mask_sum = torch.clamp(mask.sum(), min=1.)
                    
                    att_from_att_loss = self.criterion(att_from_att, batch[0]).mean()
                    att_from_weight_loss = (self.criterion(att_from_weight, batch[0])*mask).sum(0).mean()/mask_sum
                    if self.opt.single_modal_ablation:
                        att_from_weight_loss = 0 * att_from_weight_loss
                    weight_from_weight_loss = (self.criterion(weight_from_weight, batch[1])*mask).sum(0).mean()/mask_sum
                    weight_from_att_loss = (self.criterion(weight_from_att, batch[1])*mask).sum(0).mean()/mask_sum

                    loss = att_from_att_loss + att_from_weight_loss + weight_from_weight_loss + weight_from_att_loss

                    if self.opt.conclude_inv:
                        if self.opt.concatenation:
                            weight_output_f = weight_from_att[:batch[2].shape[0],:]
                            weight_output_inv = weight_from_att[batch[2].shape[0]:,:]
                            seperate_loss = - self.criterion(weight_output_f, weight_output_inv).mean()
                            if seperate_loss > self.opt.sep_param:
                                if self.opt.seperate_loss:
                                    loss = loss + seperate_loss
                        else:
                            weight_from_att_inv_loss = - self.criterion(weight_from_att, weight_from_att_inv).mean()
                            if weight_from_att_inv_loss > self.opt.sep_param:
                                if self.opt.seperate_loss:
                                    loss = loss + weight_from_att_inv_loss
                    
                    if self.opt.subspace_proj and not self.opt.concatenation:
                        mut = weight_from_att @ self.Q
                        mutnorm = mut / torch.norm(self.Q.T, dim=1).unsqueeze(0)
                        proj_weights = mutnorm @ self.Q.T
                        proj_weights = proj_weights.squeeze()
                        subspace_proj_loss = 0.001 * torch.norm(weight_from_att - proj_weights)
                        loss += subspace_proj_loss

                epoch_loss += loss.data

                loss.backward()
                self.weight_optimizer.step()

            epoch_loss /= len(self.loader)
            epoch_losses.append(epoch_loss)
            if epoch == 0:
                prev_loss = epoch_loss
            else:
                loss_diff = torch.abs(prev_loss - epoch_loss)
                prev_loss = epoch_loss

            epoch_info = {"loss": epoch_loss}

            slope = torch.tensor(0)
            if self.opt.early_stopping_slope:
                if epoch > 20:
                    threshold = 2 * 10e-4 if self.opt.cos_sim_loss else 2 * 10e-7
                    slope = - (torch.mean(torch.stack(epoch_losses)[-10:]) - torch.mean(torch.stack(epoch_losses)[-20:-10])) / 10.
                    if slope < threshold and slope > 0:
                        counter += 1
                        if counter == 5:
                            breaking = True
                    else:
                        counter = 0
                    epoch_info["slope"] = slope.item()

            if self.opt.conclude_inv:
                file_path = f"{self.opt.rootpath}/log/{self.opt.dataset}/{self.opt.llm}_loss_inv.txt"
            else:
                file_path = f"{self.opt.rootpath}/log/{self.opt.dataset}/{self.opt.llm}_loss.txt"
            with open(file_path, 'a', encoding='utf-8') as file_object:
                if slope.item():
                    file_object.write(f"Loss:{epoch_loss.item()}, slope:{slope.item()}\n")
                else:
                    file_object.write(f"Loss:{epoch_loss.item()}\n")

            # Calculate the performance (ZSL or GZSL) of weights predicted.
            if ((not self.opt.strict_eval) or (epoch + 1 == self.nepoch) or breaking) and not self.opt.daegnn:
                self.model.eval()
                if epoch + 1 == self.nepoch or breaking:
                    self.calc_entropy = self.opt.calc_entropy

                val_out = self.pred_weights_and_val(weight_model=self.model)
                self.model.train()

                if self.opt.zst:
                    acc_target, acc_zst_unseen = val_out
                else:
                    acc_gzsl, acc_seen, acc_unseen, H, acc_unseen_zsl = val_out

                    epoch_info["acc_unseen_zsl"] = acc_unseen_zsl
                    epoch_info["H"] = H
                    epoch_info["acc_unseen_gzsl"] = acc_unseen
                    epoch_info["acc_seen_gzsl"] = acc_seen

                    if H >= run_best_H:
                        run_best_acc_gzsl, run_best_acc_seen, run_best_acc_unseen, run_best_H, run_best_unseen_zsl = acc_gzsl, acc_seen, acc_unseen, H, acc_unseen_zsl

            if breaking:
                print("Stopping early (slope criterion)")
                break
        
        # baseline wDAE
        if self.opt.daegnn:
            print("Starting training of wDAE-GNN")
            comp_loss = nn.MSELoss(reduction='none')
            counter = 0
            breaking = False
            epoch_losses = []
            for epoch in trange(self.nepoch):
                epoch_loss = 0
                for _, batch in enumerate(self.dae_loader):
                    self.model.zero_grad()
                    self.dae.zero_grad()
                    
                    if self.opt.single_autoencoder_baseline:
                        att, weights = batch
                        output = self.model(att).detach()
                        perm = torch.randperm(weights.size(0))
                        weights_input = weights.unsqueeze(0).repeat(self.dae_meta_batch_size, 1, 1)

                        num_idxs = weights.size(0) // self.dae_meta_batch_size
                        for i in range(self.dae_meta_batch_size):
                            idx = perm[i*num_idxs:(i+1)*num_idxs]
                            weights_input[i][idx] = output[idx]

                        recon = self.dae(weights_input)
                        loss = comp_loss(recon, weights_input).mean()
                        loss.backward()
                        self.dae_optimizer.step()
                    else:
                        att, weights = batch
                        output = self.model(batch)
                        att_from_att, att_from_weight, weight_from_weight, weight_from_att, latent_att, latent_weight = output
                        perm = torch.randperm(weights.size(0))
                        weights_input = weights.unsqueeze(0).repeat(self.dae_meta_batch_size, 1, 1)

                        num_idxs = weights.size(0) // self.dae_meta_batch_size
                        for i in range(self.dae_meta_batch_size):
                            idx = perm[i*num_idxs:(i+1)*num_idxs]
                            weights_input[i][idx] = weight_from_att[idx]

                        recon = self.dae(weights_input)
                        loss = comp_loss(recon, weights_input).mean()
                        loss.backward()
                        self.dae_optimizer.step()

                    epoch_loss += loss.data

                epoch_loss /= len(self.dae_loader)
                epoch_losses.append(epoch_loss)
                if epoch == 0:
                    prev_loss = epoch_loss
                else:
                    loss_diff = torch.abs(prev_loss - epoch_loss) 
                    prev_loss = epoch_loss
                
                if self.opt.single_autoencoder_baseline:
                    epoch_info = {"loss": epoch_loss}

                if self.opt.early_stopping_slope:
                    if epoch > 20:
                        threshold = 2 * 10e-4 if self.opt.cos_sim_loss else 2 * 10e-7 
                        slope = - (torch.mean(torch.stack(epoch_losses)[-10:]) - torch.mean(torch.stack(epoch_losses)[-20:-10])) / 10.
                        if slope < threshold:
                            counter += 1
                            if counter == 5:
                                breaking = True
                        else:
                            counter = 0
                        epoch_info["slope"] = slope 

                if (not self.opt.strict_eval) or (epoch + 1 == self.nepoch) or breaking:
                    self.model.eval()
                    self.dae.eval()
                    if epoch + 1 == self.nepoch or breaking:
                        self.calc_entropy = self.opt.calc_entropy
                    
                    val_out = self.pred_weights_and_val(weight_model=self.model, daegnn=self.dae)
                    self.model.train()
                    self.dae.train()

                    if self.opt.zst:
                        acc_target, acc_zst_unseen = val_out
                    else:
                        acc_gzsl, acc_seen, acc_unseen, H, acc_unseen_zsl = val_out

                        epoch_info["acc_unseen_zsl"] = acc_unseen_zsl
                        epoch_info["H"] = H
                        epoch_info["acc_unseen_gzsl"] = acc_unseen
                        epoch_info["acc_seen_gzsl"] = acc_seen
                            
                        if H >= run_best_H:
                            print("New best GZSL based on H (seed):", H)
                            run_best_acc_gzsl, run_best_acc_seen, run_best_acc_unseen, run_best_H, run_best_unseen_zsl = acc_gzsl, acc_seen, acc_unseen, H, acc_unseen_zsl

                if breaking:
                    print("Stopping early")
                    break

        if self.opt.zst:
            self.acc_target, self.acc_zst_unseen = acc_target, acc_zst_unseen
        else:
            self.acc_gzsl, self.acc_seen, self.acc_unseen, self.H, self.acc_unseen_zsl = run_best_acc_gzsl, run_best_acc_seen, run_best_acc_unseen, run_best_H, run_best_unseen_zsl
