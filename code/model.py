# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix, lil_matrix

class DRDM(nn.Module):
    def __init__(self, config, dataset):
        super(DRDM, self).__init__()
        train_manager, train_adj, disease_adj, drug_adj, pos_weight = dataset
        self.interaction_matrix = coo_matrix(train_adj).astype(np.float32)
        self.disease_adj = disease_adj
        self.drug_adj = drug_adj
        self.dr_adj = torch.sparse.FloatTensor(
            torch.LongTensor([self.interaction_matrix.row, self.interaction_matrix.col]),
            torch.FloatTensor(self.interaction_matrix.data),
            torch.Size(self.interaction_matrix.shape)
        )
        self.rd_adj = self.dr_adj.transpose(0, 1)
        self.path = './dataset/' + str(config['dataset'])
        self.device = config['device']
        self.exp_coff = config['exp_coff']
        self.n_diseases = self.interaction_matrix.shape[0]
        self.n_drugs = self.interaction_matrix.shape[1]
        self.Graph = self.getSparseGraph()
        self.Graph2 = self.getSparseGraph2()
        self.dropout = nn.Dropout(p=0.1, inplace=True)
        self.pos_weight = pos_weight
        self.batch_size = config['batch_size']
        self.latent_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.disease_embedding = torch.nn.Embedding(num_embeddings=self.n_diseases, embedding_dim=self.latent_dim)
        self.drug_embedding = torch.nn.Embedding(num_embeddings=self.n_drugs, embedding_dim=self.latent_dim)
        self.wd = config['wd']
        self.wr = config['wr']

        # self-gating
        self.gating_weightdb = nn.Parameter(
            torch.FloatTensor(1, self.latent_dim))
        nn.init.xavier_normal_(self.gating_weightdb.data)
        self.gating_weightd = nn.Parameter(
            torch.FloatTensor(self.latent_dim, self.latent_dim))
        nn.init.xavier_normal_(self.gating_weightd.data)
        self.gating_weightrb = nn.Parameter(
            torch.FloatTensor(1, self.latent_dim))
        nn.init.xavier_normal_(self.gating_weightrb.data)
        self.gating_weightr = nn.Parameter(
            torch.FloatTensor(self.latent_dim, self.latent_dim))
        nn.init.xavier_normal_(self.gating_weightr.data)

    def self_gatingd(self, em):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.gating_weightd) + self.gating_weightdb))

    def self_gatingr(self, em):
        return torch.multiply(em, torch.sigmoid(torch.matmul(em, self.gating_weightr) + self.gating_weightrb))

    def get_weighted_adj_emb(self, adj, emb):
        indices = adj._indices()
        values = adj._values()
        if self.training:
            sign = torch.sign(emb)
            random_noise = nn.functional.normalize(torch.rand(emb.shape).to(self.device)) * 0.1
            emb = emb + sign * random_noise
        start_emb = (emb[indices[0]]).detach()
        end_emb = (emb[indices[1]])
        cross_product = (torch.mul(start_emb, end_emb).mean(dim=1))
        mat = (2 - 2 * cross_product) / self.exp_coff
        mat = mat * values
        new_indices = indices[0].unsqueeze(1).expand(end_emb.shape)
        weighted_emb = torch.mul(self.dropout(end_emb), mat.unsqueeze(1).expand(end_emb.shape))
        update_all_emb = torch.zeros(emb.shape).to(self.device)
        update_all_emb.scatter_add_(0, new_indices, weighted_emb)
        return update_all_emb

    def getSparseGraph(self):
        adj_mat = sp.dok_matrix((self.n_diseases + self.n_drugs, self.n_diseases + self.n_drugs), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()
        adj_mat[:self.n_diseases, self.n_diseases:] = R
        adj_mat[self.n_diseases:, :self.n_diseases] = R.T
        adj_mat = adj_mat.todok()
        rowsum = np.array(adj_mat.sum(axis=1))
        rowsum[rowsum == 0.] = 1.
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        Graph = Graph.coalesce().to(self.device)
        return Graph

    def getSparseGraph2(self):
        adj_mat = sp.dok_matrix((self.n_diseases + self.n_drugs, self.n_diseases + self.n_drugs), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R1 = lil_matrix(self.disease_adj)
        R2 = lil_matrix(self.drug_adj)
        adj_mat[:self.n_diseases, :self.n_diseases] = R1
        adj_mat[self.n_diseases:, self.n_diseases:] = R2
        adj_mat = adj_mat.todok()
        rowsum = np.array(adj_mat.sum(axis=1))
        rowsum[rowsum == 0.] = 1.
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)
        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        norm_adj = norm_adj.tocsr()
        Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
        Graph = Graph.coalesce().to(self.device)
        return Graph

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def get_ego_embeddings(self):
        r"""Get the embedding of diseases and drugs and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_drugs+n_diseases, embedding_dim]
        """
        disease_embeddings = self.disease_embedding.weight
        drug_embeddings = self.drug_embedding.weight
        dd_embeddings = self.self_gatingd(disease_embeddings)
        rr_embeddings = self.self_gatingr(drug_embeddings)
        ego_embeddings = torch.cat([disease_embeddings, drug_embeddings], dim=0)
        ego_gating_embeddings = torch.cat([dd_embeddings, rr_embeddings], dim=0)
        return ego_embeddings, ego_gating_embeddings

    # Contrastive Learning
    def ssl_loss(self, data1, data2, index):
        index = torch.unique(torch.LongTensor(index))
        embeddings1 = data1[index]
        embeddings2 = data2[index]
        norm_embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        norm_embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        pos_score = torch.sum(torch.mul(norm_embeddings1, norm_embeddings2), dim=1)
        all_score = torch.mm(norm_embeddings1, norm_embeddings2.T)
        pos_score = torch.exp(pos_score / 0.05)
        all_score = torch.sum(torch.exp(all_score / 0.05), dim=1)
        ssl_loss = (-torch.sum(torch.log(pos_score / ((all_score)))) / (len(index)))
        return ssl_loss

    def forward(self, batch, is_training):
        diseases, drugs, labels = batch
        all_embeddings, all_gating_embeddings = self.get_ego_embeddings()
        d_r_embeddings_list = [all_embeddings]
        dd_rr_embeddings_list = [all_gating_embeddings]
        all_emb = all_embeddings
        all_emb_gating = all_gating_embeddings
        for layer_idx in range(1, self.n_layers + 1):
            all_emb = self.get_weighted_adj_emb(self.Graph, all_emb)
            all_emb_gating = self.get_weighted_adj_emb(self.Graph2, all_emb_gating)
            all_embeddings1 = all_emb
            all_embeddings2 = all_emb_gating
            d_r_embeddings_list.append(all_embeddings1)
            dd_rr_embeddings_list.append(all_embeddings2)
        d_r_embeddings = torch.stack(d_r_embeddings_list, dim=1)
        d_r_embeddings = torch.mean(d_r_embeddings, dim=1)
        dd_rr_embeddings = torch.stack(dd_rr_embeddings_list, dim=1)
        dd_rr_embeddings = torch.mean(dd_rr_embeddings, dim=1)
        dr_diseaseEmbedding, dr_drugEmbedding = torch.split(d_r_embeddings, [self.n_diseases, self.n_drugs])
        diseaseEmbedding, drugEmbedding = torch.split(dd_rr_embeddings, [self.n_diseases, self.n_drugs])
        self.dr_diseaseEmbedding = dr_diseaseEmbedding
        self.dr_drugEmbedding = dr_drugEmbedding
        self.diseaseEmbedding = diseaseEmbedding
        self.drugEmbedding = drugEmbedding
        self.fuse_diseaseEmbedding = (self.wd * self.dr_diseaseEmbedding + (1 - self.wd) * self.diseaseEmbedding)
        self.fuse_drugEmbedding = (self.wr * self.dr_drugEmbedding + (1 - self.wr) * self.drugEmbedding)
        batch_disease_all_embeddings = self.fuse_diseaseEmbedding[diseases]
        batch_drug_all_embeddings = self.fuse_drugEmbedding[drugs]
        scores = torch.mul(batch_disease_all_embeddings, batch_drug_all_embeddings).sum(dim=1)
        labels = torch.FloatTensor(labels)
        BCE_loss = self.bce_loss_fn(torch.sigmoid(scores), labels)
        # Contrastive Learning of collaborative relations
        ssl_loss_disease = self.ssl_loss(self.dr_diseaseEmbedding, self.diseaseEmbedding, diseases)
        ssl_loss_drug = self.ssl_loss(self.dr_drugEmbedding, self.drugEmbedding, drugs)
        ssl_loss = 0.05 * ssl_loss_disease + 0.05 * ssl_loss_drug
        loss = BCE_loss + 0.3 * ssl_loss
        return loss, torch.sigmoid(scores)

    def predict(self, batch):
        _, _, labels = batch
        _, scores = self.forward(batch, False)
        return scores, labels

    def bce_loss_fn(self, predict, label):
        predict = predict.reshape(-1)
        label = label.reshape(-1)
        weight = self.pos_weight * label + 1 - label
        # if you use cpu:
        loss = F.binary_cross_entropy(input=predict, target=label, weight=weight)
        # if you use gpu:
        # loss = F.binary_cross_entropy(input=predict.to("cuda:0"), target=label.to("cuda:0"), weight=weight.to("cuda:0"))
        return loss