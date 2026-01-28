import torch.nn as nn
import torch
from diffurec import DiffuRec
import torch.nn.functional as F
import torch as th
import random

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Att_Diffuse_model(nn.Module):
    def __init__(self, diffu, args):
        super(Att_Diffuse_model, self).__init__()
        self.emb_dim = args.hidden_size
        self.item_num = args.item_num+1
        self.p = args.p
        self.batch_size = args.batch_size
        self.souce_embeddings = nn.Embedding(args.source_item_num+1, self.emb_dim)
        self.target_embeddings = nn.Embedding(args.target_item_num+1, self.emb_dim)
        self.shared_layer = nn.Linear(self.emb_dim, self.emb_dim)
        self.specific_layer = nn.Linear(self.emb_dim, self.emb_dim)
        self.embed_dropout = nn.Dropout(args.emb_dropout)
        self.position_embeddings = nn.Embedding(args.max_len, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.dropout)
        self.diffu = diffu
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_ce_rec = nn.CrossEntropyLoss(reduction='none')
        self.loss_mse = nn.MSELoss()
       
    def diffu_pre(self, item_rep, tag_emb, mask_seq):
        seq_rep_diffu, item_rep_out, weights, t  = self.diffu(item_rep, tag_emb, mask_seq)
        return seq_rep_diffu, item_rep_out, weights, t

    def reverse(self, item_rep, item_rep1, noise_x_t, mask_seq):
        reverse_pre = self.diffu.reverse_p_sample(item_rep, item_rep1, noise_x_t, mask_seq)
        return reverse_pre

    def loss_rec(self, scores, labels):
        return self.loss_ce(scores, labels.squeeze(-1))


    def loss_diffu_ce(self, rep_diffu, labels, pretrain_flag):
        if pretrain_flag:
            scores = torch.matmul(rep_diffu, self.shared_layer(self.souce_embeddings.weight).t())
        else:
            scores = torch.matmul(rep_diffu, self.target_embeddings.weight.t())
        return self.loss_ce(scores, labels.squeeze(-1))

    def diffu_rep_pre(self, rep_diffu, pretrain_flag):
        if pretrain_flag:
            scores = torch.matmul(rep_diffu, self.shared_layer(self.souce_embeddings.weight).t())
        else:
            scores = torch.matmul(rep_diffu, self.target_embeddings.weight.t())
        return scores

    def regularization_rep(self, seq_rep, mask_seq):
        seqs_norm = seq_rep/seq_rep.norm(dim=-1)[:, :, None]
        seqs_norm = seqs_norm * mask_seq.unsqueeze(-1)
        cos_mat = torch.matmul(seqs_norm, seqs_norm.transpose(1, 2))
        cos_sim = torch.mean(torch.mean(torch.sum(torch.sigmoid(-cos_mat), dim=-1), dim=-1), dim=-1)  ## not real mean
        return cos_sim

    def regularization_seq_item_rep(self, seq_rep, item_rep, mask_seq):
        item_norm = item_rep/item_rep.norm(dim=-1)[:, :, None]
        item_norm = item_norm * mask_seq.unsqueeze(-1)

        seq_rep_norm = seq_rep/seq_rep.norm(dim=-1)[:, None]
        sim_mat = torch.sigmoid(-torch.matmul(item_norm, seq_rep_norm.unsqueeze(-1)).squeeze(-1))
        return torch.mean(torch.sum(sim_mat, dim=-1)/torch.sum(mask_seq, dim=-1))


    def embedding_loss(self, item_shared_embeddings, con_shared_embeddings, 
                    item_specific_embeddings, con_specific_embeddings, 
                    lambda_shared=200.0, lambda_private=0.2, lambda_orthogonal=20):
       
        item_orthogonal_loss = torch.mean(F.cosine_similarity(item_shared_embeddings, item_specific_embeddings, dim=-1) ** 2)
        con_orthogonal_loss = torch.mean(F.cosine_similarity(con_shared_embeddings, con_specific_embeddings, dim=-1) ** 2)
        orthogonal_loss = item_orthogonal_loss + con_orthogonal_loss

        loss_mmd_shared = self.mmd_loss(item_shared_embeddings,con_shared_embeddings)
        loss_mmd_private =1/self.mmd_loss(item_specific_embeddings, con_specific_embeddings)

        total_loss = lambda_shared * loss_mmd_shared + lambda_private * loss_mmd_private + lambda_orthogonal * orthogonal_loss
        
        return total_loss

    def rbf_kernel(self, x, y, sigma=1.0, chunk_size=4096):
        device = x.device
        batch_size_x = x.size(0)
        batch_size_y = y.size(0)
        x_norm = (x ** 2).sum(dim=1, keepdim=True)  
        y_norm = (y ** 2).sum(dim=1, keepdim=True)

        mean_kernel_value = 0.0
        total_elements = 0

        for i in range(0, batch_size_x, chunk_size):
            x_chunk = x[i:i + chunk_size]
            x_chunk_norm = x_norm[i:i + chunk_size]

            for j in range(0, batch_size_y, chunk_size):
                y_chunk = y[j:j + chunk_size]
                y_chunk_norm = y_norm[j:j + chunk_size]

                dist_chunk = x_chunk_norm - 2 * th.matmul(x_chunk, y_chunk.T) + y_chunk_norm.T
                kernel_chunk = th.exp(-dist_chunk / (2 * sigma ** 2)) 

                mean_kernel_value += kernel_chunk.sum()
                total_elements += kernel_chunk.numel()

        mean_kernel_value /= total_elements
        return mean_kernel_value
    
    def mmd_loss(self,x, y, sigma=1.0):
        x = x.mean(dim=1)
        y = y.mean(dim=1)
        k_xx = self.rbf_kernel(x, x, sigma)  
        k_yy = self.rbf_kernel(y, y, sigma)  
        k_xy = self.rbf_kernel(x, y, sigma)  
        mmd_loss = k_xx+ k_yy - 2 * k_xy
        return mmd_loss

    def forward(self, sequence, tag, con_seq, pretrain_flag, args, epoch, train_flag=True): 
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        position_embeddings = self.position_embeddings(position_ids)
        mask_seq = (sequence>0).float()
        device = args.device
        em_loss = 0

        if pretrain_flag:
            item_embeddings = self.souce_embeddings(sequence)
            item_embeddings = item_embeddings + position_embeddings
            item_shared_embeddings = self.shared_layer(item_embeddings)
            item_embeddings = item_shared_embeddings
            item_specific_embeddings = item_embeddings - item_shared_embeddings
            
            if train_flag:
                
               
                con_embeddings= self.target_embeddings(con_seq)
                con_embeddings = con_embeddings + position_embeddings

                con_shared_embeddings = self.shared_layer(con_embeddings)
                con_specific_embeddings = con_embeddings - con_shared_embeddings
                em_loss = self.embedding_loss(item_shared_embeddings, con_shared_embeddings, item_specific_embeddings, con_specific_embeddings)
        
        else:
            item_embeddings = self.target_embeddings(sequence)
            item_embeddings = item_embeddings + position_embeddings
            if train_flag:
                item_shared_embeddings = self.shared_layer(item_embeddings)
                num = random.random()
                
                if num < self.p:
                    item_embeddings = item_shared_embeddings
            else:
                item_shared_embeddings = self.shared_layer(item_embeddings)
                item_shared_embeddings = self.embed_dropout(item_shared_embeddings)
                item_shared_embeddings = self.LayerNorm(item_shared_embeddings)
            
        item_embeddings = self.embed_dropout(item_embeddings)  

        item_embeddings = self.LayerNorm(item_embeddings)
        
        
        if train_flag:
            if pretrain_flag:
                tag_emb = self.shared_layer(self.souce_embeddings(tag.squeeze(-1))) 
            else:
                tag_emb = self.target_embeddings(tag.squeeze(-1))
 
            rep_diffu, rep_item, weights, t = self.diffu(item_embeddings, tag_emb, mask_seq)

            
            item_rep_dis = None
            seq_rep_dis = None
        else:
            noise_x_t = th.randn_like(item_embeddings[:,-1,:])
            if pretrain_flag:
                rep_diffu = self.reverse(item_embeddings, item_embeddings, noise_x_t, mask_seq)
            else:
                rep_diffu = self.reverse(item_embeddings, item_shared_embeddings, noise_x_t, mask_seq)
            weights, t, item_rep_dis, seq_rep_dis = None, None, None, None

        scores = None
        return scores, rep_diffu, weights, t, item_rep_dis, seq_rep_dis, em_loss
        

def create_model_diffu(args):
    diffu_pre = DiffuRec(args)
    return diffu_pre
