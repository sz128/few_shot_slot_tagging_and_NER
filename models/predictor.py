#!/usr/bin/env python3

'''
@Time   : 2020-05-05 21:26:14
@Author : su.zhu
@Desc   : 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
import math

MIN_NUMBER = -1e32 # -float('inf')
EPS = 1e-16

def get_unit_vectors(x, eps=EPS):
    """ x : B * L * H or B * H """
    x_l2_norm = torch.norm(x, p=2, dim=-1, keepdim=True) + eps
    x = x.div(x_l2_norm.expand_as(x))
    return x

class Projection(nn.Module):

    def __init__(self, config, input_embedding_size, projection=None, projection_dim=0, device=None):
        '''
        projection: 
            adaptive_finetune
            LSM (least squares method)
            LEN (linear error nulling)
        '''
        super().__init__()
        
        self.device = device
        self.input_embedding_size = input_embedding_size
        self.projection = projection
        self.projection_dim = projection_dim
        if self.projection_dim == 0 or projection is None:
            self.projection_dim = input_embedding_size

        if self.projection == 'learnable':
            self.M = nn.Linear(input_embedding_size, self.projection_dim, bias=False)
        elif self.projection == None:
            self.M = nn.Identity()
        elif self.projection == 'adaptive_finetune' or self.projection == 'LSM':
            self.M = nn.Linear(input_embedding_size, self.projection_dim, bias=False)
            self.M.weight.requires_grad = False
        elif self.projection == 'LEN':
            self.M = None
        else:
            exit()

    def reset(self):
        self.M.weight.data.normal_(mean=0.0, std=0.02)

    def get_output_dim(self):
        return self.projection_dim

    def construction_of_projection_M_LSM(self, C, X, Y):
        '''
        C : M * H, per-class average of the embedded features
        X : N * H
        Y : N
        '''
        #assert self.projection == 'LSM'
        lambda_ = 1e-2
        m, h = C.size()
        n = X.size()[0]
        left = torch.mm(X.t(), X) + lambda_ * torch.eye(h, device=self.device)
        left = left.inverse()
        left = torch.mm(left, X.t())
        right = torch.mm(C.t(), C) + lambda_ * torch.eye(h, device=self.device)
        right = right.inverse()
        right = torch.mm(C, right)
        Y = torch.eye(m, device=self.device).index_select(0, Y)
        Y -= 0.5
        Y *= 2
        M = torch.mm(left, Y)
        M = torch.mm(M, right)
        self.M.weight.data = M

    def construction_of_projection_M_LEN(self, Phi, C):
        '''
        Phi : n * p
        C : n * p, b * n * p, per-class average of the embedded features
        '''
        eps = 1e-6
        c_shape = C.shape
        if len(c_shape) == 2:
            n, p = c_shape
            Phi_sum = Phi.sum(dim=0)
            mod_Phi = (n * Phi - Phi_sum[None, :]) / (n - 1)
            mod_Phi = get_unit_vectors(mod_Phi, eps=eps)
            C = get_unit_vectors(C, eps=eps)
            null = mod_Phi - C
            
            tol = 1e-13
            u, s, vh = torch.svd(null, some=False)
            d = (s >=  tol).sum().item()
            #d = min(p - n, d)
            #M = vh[d:].conj()
            M = vh[d:].t()
        else:
            b, n, p = c_shape
            Phi_sum = Phi.sum(dim=0)
            mod_Phi = (n * Phi - Phi_sum[None, :]) / (n - 1)
            mod_Phi = get_unit_vectors(mod_Phi, eps=eps)
            C = get_unit_vectors(C, eps=eps)
            null = mod_Phi[None, :, :] - C
            
            tol = 1e-13
            u, s, vh = torch.svd(null, some=False)
            d = n
            M = vh[:, d:, :].transpose(-1, -2)
        self.M = M

    def forward(self, X):
        """
        X : 
            1. B * L * H or B * H or O * H  x  H * H'
            1. B * L * H or B * H or B * O * H  x  B * H * H'
        """
        if self.projection == 'LEN':
            if len(X.shape) == 2:
                X = X[:, None, :]
                squeeze = True
            else:
                squeeze = False
            X = torch.matmul(X, self.M)
            if squeeze:
                X = X.squeeze(1) # B * 1 * H' => B * H'
        else:
            X = self.M(X)
        return X

class MatchingClassifier(nn.Module):

    def __init__(self, config, matching_similarity_function='dot'):
        super().__init__()

        self.matching_similarity_function = matching_similarity_function.lower()

    def forward(self, encoder_hiddens, label_embeddings, masked_output=None):
        """
        encoder_hiddens : B * L * H or B * H
        label_embeddings : O * H or B * O * H
        """
        x = encoder_hiddens
        y = label_embeddings

        if len(x.shape) == 2:
            x = x[:, None, :]
            squeeze = True
        else:
            squeeze = False

        # dot product
        if self.matching_similarity_function == 'dot':
            logits = torch.matmul(x, y.transpose(-1, -2))
        elif self.matching_similarity_function == 'euclidean':
            if len(y.shape) == 3:
                logits = torch.matmul(x, y.transpose(-1, -2)) - (torch.norm(y, p=2, dim=-1) ** 2)[:, None, :] / 2
            else:
                logits = torch.matmul(x, y.transpose(-1, -2)) - (torch.norm(y, p=2, dim=-1) ** 2) / 2
        elif self.matching_similarity_function == 'euclidean2':
            norm_y = torch.norm(y, p=2, dim=-1)
            if len(y.shape) == 3:
                unit_y = y / norm_y[:, :, None].expand_as(y)
                logits = torch.matmul(x, unit_y.transpose(-1, -2)) - norm_y[:, None, :] / 2
            else:
                unit_y = y / norm_y[:, None].expand_as(y)
                logits = torch.matmul(x, unit_y.transpose(-1, -2)) - norm_y / 2
        elif self.matching_similarity_function == 'euclidean3':
            norm_y = torch.norm(y, p=2, dim=-1)
            if len(y.shape) == 3:
                unit_y = y / norm_y[:, :, None].expand_as(y)
                logits = torch.matmul(x, unit_y.transpose(-1, -2)) - norm_y[:, None, :] / 2 - (torch.norm(x, p=2, dim=-1) ** 2)[:, :, None] / norm_y[:, None, :] / 2
            else:
                unit_y = y / norm_y[:, None].expand_as(y)
                logits = torch.matmul(x, unit_y.transpose(-1, -2)) - norm_y / 2 - (torch.norm(x, p=2, dim=-1) ** 2)[:, :, None] / norm_y[None, None, :] / 2
        else:
            exit()

        if squeeze:
            logits = logits.squeeze(1) # B * 1 * O => B * O
        if masked_output is not None:
            logits = logits.index_fill_(-1, masked_output, MIN_NUMBER)

        return logits

