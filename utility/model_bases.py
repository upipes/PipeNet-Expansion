import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class LINEAR(nn.Module):
    def __init__(self, input_dim, nclass, bias=True):
        super(LINEAR, self).__init__()
        self.fc = nn.Linear(input_dim, nclass, bias)

    def forward(self, x):
        o = self.fc(x)
        return o


class LINEAR_TO_COS_SIM(nn.Module):
    def __init__(self, weights):
        super(LINEAR, self).__init__()
        self.weights = weights
        self.cos = nn.functional.cosine_similarity(dim=1)

    def forward(self, x):
        out = []
        for sample in x:
            temp = []
            for weight in self.weights:
                temp.append(self.cos(weight, sample))
            out.append(torch.stack(temp))
        o = torch.stack(out)
        return o

class ATT_AUTOENCODER_inv(nn.Module):
    def __init__(self, opt, input_dim, embed_dim, att_dim=312, output_dim=None, wordemb_dim=512):
        super(ATT_AUTOENCODER_inv, self).__init__()
        self.opt = opt
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.att_dim = att_dim
        self.output_dim = output_dim
        self.wordemb_dim = wordemb_dim
        self.attention_dim = 2048

        self.embed_dim = [embed_dim, embed_dim]
        if output_dim is None:
            self.output_dim = input_dim
        self.encoder_merge = nn.Sequential(
            nn.Linear(self.att_dim+self.wordemb_dim, self.embed_dim[0]),
            nn.ReLU(inplace=True)
        )
        self.encoder_merge1 = nn.Sequential(
            nn.Linear(self.att_dim+self.attention_dim, self.embed_dim[0]),
            nn.ReLU(inplace=True)
        )
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.embed_dim[0]),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim[1], 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.output_dim)
        )

        self.attention_fc = nn.Linear(self.wordemb_dim, 1)
        self.beta = nn.Parameter(torch.tensor(0.2))
        if self.opt.conclude_inv:
            self.inv_attention = nn.Linear(self.wordemb_dim, self.attention_dim)
        if self.opt.dataset == "Road":
            self.q_proj = nn.Linear(self.wordemb_dim, self.attention_dim)
            self.k_proj = nn.Linear(self.wordemb_dim, self.attention_dim)
            self.v_proj = nn.Linear(self.wordemb_dim, self.attention_dim)
            self.out_proj = nn.Linear(self.attention_dim, self.attention_dim)

        if not self.opt.inv_merge:
            self.q_proj_inv = nn.Linear(self.wordemb_dim, self.attention_dim)
            self.k_proj_inv = nn.Linear(self.wordemb_dim, self.attention_dim)
            self.v_proj_inv = nn.Linear(self.wordemb_dim, self.attention_dim)
            self.out_proj_inv = nn.Linear(self.attention_dim, self.attention_dim)

    def scfa_pooling(self, gpt_emb):
        gpt_emb = gpt_emb.permute(0, 2, 1)
        attn_weights = self.attention_fc(gpt_emb).squeeze(-1)
        attn_weights = F.softmax(attn_weights, dim=1)
        weighted_gpt_emb = torch.sum(gpt_emb * attn_weights.unsqueeze(-1), dim=1)
        return weighted_gpt_emb

    def cross_attention(self, query, context):
        q = self.q_proj(query) 
        k = self.k_proj(context) 
        v = self.v_proj(context)  
        attn_scores = torch.bmm(q, k.transpose(1, 2))
        scale_factor = math.sqrt(self.attention_dim)
        attn_scores = attn_scores / scale_factor
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.bmm(attn_weights, v)
        output = self.out_proj(output)
        output = output.mean(dim=1)
        return output
    
    def cross_attention_inv(self, query, context):
        q = self.q_proj_inv(query) 
        k = self.k_proj_inv(context) 
        v = self.v_proj_inv(context)  
        attn_scores = torch.bmm(q, k.transpose(1, 2))
        scale_factor = math.sqrt(self.attention_dim)
        attn_scores = attn_scores / scale_factor
        attn_weights = F.softmax(attn_scores, dim=-1)
        output = torch.bmm(attn_weights, v)
        output = self.out_proj_inv(output)
        output = output.mean(dim=1)
        return output

    def encode(self, x, x_inv=None, flag=False):
        # split two parts
        self.attribute_f = x
        # factual representation
        att_emb = self.attribute_f[:, :self.att_dim]
        desc_emb = self.attribute_f[:, self.att_dim:self.att_dim+self.wordemb_dim]
        gpt_emb = self.attribute_f[:, self.att_dim+self.wordemb_dim:].view(x.shape[0], -1, self.wordemb_dim)

        self.attribute_inv = x_inv
        # intervention representation
        if self.opt.inv_merge:
            att_emb_inv = self.attribute_inv[:, :self.att_dim]
            desc_emb_inv = self.attribute_inv[:, self.att_dim:self.att_dim+self.wordemb_dim]
        else:
            att_emb_inv = self.attribute_inv[:, :self.att_dim]
            desc_emb_inv = self.attribute_inv[:, self.att_dim:].view(x.shape[0], -1, self.wordemb_dim)
        
        # factual branch
        if self.opt.factual_branch == 'attention':
            fused_context = self.cross_attention(desc_emb_inv, gpt_emb)
            x = torch.cat([att_emb, fused_context], dim=1)
            if not flag:
                if self.opt.inv_merge:
                    desc_emb_inv = self.inv_attention(desc_emb_inv)
                    x_inv = torch.cat([att_emb_inv, desc_emb_inv], dim=1)
                elif self.opt.intervention_branch == 'attention':
                    x_inv = self.cross_attention_inv(gpt_emb, desc_emb_inv)
                    x_inv = torch.cat([att_emb_inv, x_inv], dim=1)
                elif self.opt.intervention_branch == 'mean':
                    desc_emb_inv = self.inv_attention(desc_emb_inv)
                    x_inv = torch.mean(desc_emb_inv, dim=1)
                    x_inv = torch.cat([att_emb_inv, x_inv], dim=1)
                x = torch.cat([x, x_inv], dim=0)

            return self.encoder_merge1(x)

        elif self.opt.factual_branch == 'mean':
            gpt_emb_t = gpt_emb.permute(0, 2, 1)
            gpt_emb_mean = self.scfa_pooling(gpt_emb_t)

            desc_emb = self.beta*desc_emb + (1-self.beta)*gpt_emb_mean
            desc_emb = self.inv_attention(desc_emb)
            x = torch.cat([att_emb, desc_emb], dim=1)
            
            if not flag:
                if self.opt.inv_merge:
                    x_inv = torch.cat([att_emb_inv, desc_emb_inv], dim=1)
                elif self.opt.intervention_branch == 'attention':
                    x_inv = self.cross_attention_inv(gpt_emb, desc_emb_inv)
                    x_inv = torch.cat([att_emb_inv, x_inv], dim=1)
                elif self.opt.intervention_branch == 'mean':
                    x_inv = torch.mean(desc_emb_inv, dim=1)
                    x_inv = torch.cat([att_emb_inv, x_inv], dim=1)
                x = torch.cat([x, x_inv], dim=0)
            
            return self.encoder_merge1(x)
        
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x, x_inv=None, flag=True):
        z = self.encode(x, x_inv, flag)
        return self.decode(z)


class ATT_AUTOENCODER(nn.Module):
    def __init__(self, opt, input_dim, embed_dim, att_dim=312, output_dim=None, wordemb_dim=512):
        super(ATT_AUTOENCODER, self).__init__()
        self.opt = opt
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.att_dim = att_dim
        self.output_dim = output_dim
        self.wordemb_dim = wordemb_dim
        if self.opt.dataset == "SD":
            self.attention_dim = 2048
        elif self.opt.dataset == "Road":
            self.attention_dim = 2048

        self.embed_dim = [embed_dim, embed_dim]
        if output_dim is None:
            self.output_dim = input_dim
        self.encoder_merge = nn.Sequential(
            nn.Linear(self.att_dim+self.wordemb_dim, self.embed_dim[0]),
            nn.ReLU(inplace=True)
        )
        self.encoder_merge1 = nn.Sequential(
            nn.Linear(self.att_dim+self.attention_dim, self.embed_dim[0]),
            nn.ReLU(inplace=True)
        )
        if self.opt.conclude_inv:
            self.encoder_merge = nn.Sequential(
                nn.Linear((self.att_dim+self.wordemb_dim)*2, self.embed_dim[0]),
                nn.ReLU(inplace=True)
            )
            self.encoder_merge1 = nn.Sequential(
                nn.Linear((self.att_dim+self.attention_dim)*2, self.embed_dim[0]),
                nn.ReLU(inplace=True)
            )
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.embed_dim[0]),
            nn.ReLU(inplace=True)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.embed_dim[1], 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, self.output_dim)
        )
        self.attention_fc = nn.Linear(self.wordemb_dim, 1)
        self.beta = nn.Parameter(torch.tensor(0.2))

        if self.opt.conclude_inv:
            self.inv_attention = nn.Linear(self.wordemb_dim, self.attention_dim)
        self.query_layer = nn.Linear(self.wordemb_dim, self.attention_dim)
        self.query_layer_att = nn.Linear(self.att_dim, self.attention_dim)
        self.key_layer = nn.Linear(self.wordemb_dim, self.attention_dim)
        self.value_layer = nn.Linear(self.wordemb_dim, self.attention_dim)

    def scfa_pooling(self, gpt_emb):
        gpt_emb = gpt_emb.permute(0, 2, 1)
        attn_weights = self.attention_fc(gpt_emb).squeeze(-1)
        attn_weights = F.softmax(attn_weights, dim=1)
        weighted_gpt_emb = torch.sum(gpt_emb * attn_weights.unsqueeze(-1), dim=1)
        return weighted_gpt_emb

    def compute_cosed(self, q, k):
        cs = F.cosine_similarity(q, k, dim=-1)
        ed = torch.norm(q - k, p=2, dim=-1)
        cosed = cs * ed
        return cosed

    def encode(self, x):
        original_dim = self.att_dim + (self.opt.view_num + 1) * self.wordemb_dim
        # split two parts
        self.attribute_f = x[:, :original_dim]
        if self.opt.conclude_inv:
            self.attribute_inv = x[:, original_dim:]
        
        # factual representation
        att_emb = self.attribute_f[:, :self.att_dim]
        desc_emb = self.attribute_f[:, self.att_dim:self.att_dim+self.wordemb_dim]
        gpt_emb = self.attribute_f[:, self.att_dim+self.wordemb_dim:].view(x.shape[0], -1, self.wordemb_dim)

        # intervention representation
        if self.opt.conclude_inv:
            att_emb_inv = self.attribute_inv[:, :self.att_dim]
            desc_emb_inv = self.attribute_inv[:, self.att_dim:self.att_dim+self.wordemb_dim]
            gpt_emb_inv = self.attribute_inv[:, self.att_dim+self.wordemb_dim:].view(x.shape[0], -1, self.wordemb_dim)
        
        # factual branch
        if self.opt.factual_branch == 'attention':
            desc_emb_expanded = desc_emb.unsqueeze(1).expand(-1, self.opt.view_num, -1)
            q = self.query_layer(desc_emb_expanded)
            k = self.key_layer(gpt_emb)
            v = self.value_layer(gpt_emb)
            attn_scores = self.compute_cosed(q, k)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            context = torch.matmul(attn_weights, v)
            fused_context = context.mean(dim=1)
            x = torch.cat([att_emb, fused_context], dim=1)
            if self.opt.conclude_inv:
                desc_emb_inv = self.inv_attention(desc_emb_inv)
                x_inv = torch.cat([att_emb_inv, desc_emb_inv], dim=1)
                x = torch.cat([x, x_inv], dim=1)
            return self.encoder_merge1(x)

        elif self.opt.factual_branch == 'mean':
            gpt_emb = gpt_emb.permute(0, 2, 1)
            gpt_emb = self.scfa_pooling(gpt_emb)

            x = torch.cat([att_emb, self.beta*desc_emb + (1-self.beta)*gpt_emb], dim=1)
            
            if self.opt.conclude_inv:
                x_inv = torch.cat([att_emb_inv, desc_emb_inv], dim=1)
                x = torch.cat([x, x_inv], dim=1)
            
            return self.encoder_merge(x)
        
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x, x_inv=None, flag=False):
        z = self.encode(x)
        return self.decode(z)

class AUTOENCODER(nn.Module):
    def __init__(self, opt, input_dim, embed_dim, output_dim=None, num_layers=3):
        super(AUTOENCODER, self).__init__()
        self.opt = opt
        self.input_dim = input_dim
        self.output_dim = output_dim
        if output_dim is None:
            self.output_dim = input_dim
        self.embed_dim = [embed_dim, embed_dim]
        if num_layers == 2:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.embed_dim[0]),
                nn.ReLU(inplace=True)
            )

            self.decoder = nn.Sequential(
                nn.Linear(self.embed_dim[1], self.output_dim)
            )

            if self.opt.conclude_inv and not self.opt.concatenation:
                self.decoder_inv = nn.Sequential(
                    nn.Linear(self.embed_dim[1], self.output_dim)
                )
        if num_layers == 3:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.embed_dim[0]),
                nn.ReLU(inplace=True)
            )

            self.decoder = nn.Sequential(
                nn.Linear(self.embed_dim[1], 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, self.output_dim)
            )
        if num_layers == 4:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_dim, self.embed_dim[0]),
                nn.ReLU(inplace=True),
                nn.Linear(self.embed_dim[0], self.embed_dim[0]),
                nn.ReLU(inplace=True)
            )

            self.decoder = nn.Sequential(
                nn.Linear(self.embed_dim[1], 1000),
                nn.ReLU(inplace=True),
                nn.Linear(1000, self.output_dim)
            )

    def encode(self, x, x_inv=None):
        if self.opt.concatenation:
            x=torch.cat((x, x_inv), dim=0)
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)
    
    def decode_inv(self, x):
        return self.decoder_inv(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

class JOINT_AUTOENCODER(nn.Module):
    def __init__(self, opt, autoencoder1, autoencoder2):
        super(JOINT_AUTOENCODER, self).__init__()
        self.ae1 = autoencoder1
        self.ae2 = autoencoder2
        self.opt = opt

    def encode1(self, x, x_inv=None, flag=False):
        if self.opt.concatenation:
            return self.ae1.encode(x, x_inv, flag)
        else:
            return self.ae1.encode(x)

    def encode2(self, x, x_inv=None):
        if self.opt.concatenation:
            return self.ae2.encode(x, x_inv)
        else:
            return self.ae2.encode(x)

    def decode1(self, x):
        return self.ae1.decode(x)

    def decode2(self, x):
        return self.ae2.decode(x)
    
    def decode_inv(self, x):
        return self.ae2.decode_inv(x)

    def forward(self, x):
        if self.opt.concatenation:
            att_in, weight_in, att_in_inv, weight_in_inv = x
        else:
            att_in, weight_in = x

        if self.opt.concatenation:
            latent_att = self.encode1(att_in, att_in_inv)
            latent_weight = self.encode2(weight_in, weight_in_inv)
        else:
            latent_att = self.encode1(att_in)
            latent_weight = self.encode2(weight_in)

        att_from_att = self.decode1(latent_att)
        att_from_weight = self.decode1(latent_weight)

        weight_from_weight = self.decode2(latent_weight)
        weight_from_att = self.decode2(latent_att)

        if self.opt.conclude_inv and not self.opt.concatenation:
            weight_from_att_inv = self.decode_inv(latent_att)
            return att_from_att, att_from_weight, weight_from_weight, weight_from_att, weight_from_att_inv, latent_att, latent_weight
        else:
            return att_from_att, att_from_weight, weight_from_weight, weight_from_att, latent_att, latent_weight

    def predict(self, x, x_inv=None, flag=True):
        latent_att = self.encode1(x, x_inv, flag)
        return self.decode2(latent_att)
