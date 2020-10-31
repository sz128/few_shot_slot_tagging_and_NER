#!/usr/bin/env python3

'''
@Time   : 2019-08-07 14:35:28
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

class SimilarityNet(nn.Module):

    def __init__(self, config, label_embedding_dim, similarity_fc='multiply', normalized=False):
        '''
        similarity_fc : cosine / multiply / ff (feed forward)
        '''
        super(SimilarityNet, self).__init__()

        self.label_embedding_dim = label_embedding_dim
        self.similarity_fc = similarity_fc
        self.normalized = normalized
        if self.similarity_fc == 'ff':
            self.w_1 = nn.Linear(label_embedding_dim, label_embedding_dim)
            self.w_2 = nn.Linear(label_embedding_dim, label_embedding_dim)
            self.w_3 = nn.Linear(label_embedding_dim, 1)
        
    def forward(self, input_embeddings, label_embeddings):
        '''
        input_embeddings : B * H or B * L * H
        label_embeddings : O * H or B * O * H
        '''
        if len(input_embeddings.size()) == 2:
            squeeze = True
            input_embeddings = input_embeddings.unsqueeze(1) # B * 1 * H
        else:
            squeeze = False
        if self.similarity_fc == 'cosine':
            # cosine 的结果很差，即同时对两个vector做单位化根本训练不动；但是仅仅只对其中任何一个向量做单位化，则训练正常；
            input_embeddings = F.normalize(input_embeddings, p=2, dim=-1)
            label_embeddings = F.normalize(label_embeddings, p=2, dim=-1)
            scores = torch.matmul(input_embeddings, label_embeddings.transpose(-1, -2)) * 10 # 因为cosine的值域是[-1, 1]， 而softmax的输入值域是(-inf, +inf)，所以需要扩大数值范围
        elif self.similarity_fc == 'multiply':
            scores = torch.matmul(input_embeddings, label_embeddings.transpose(-1, -2))
            #scores = 2 * torch.matmul(input_embeddings, label_embeddings.transpose(-1, -2))
            #scores = 2 * torch.matmul(input_embeddings, label_embeddings.transpose(-1, -2)) - torch.norm(label_embeddings, p=2, dim=1) ** 2
            if self.normalized:
                scores = scores / math.sqrt(self.label_embedding_dim)
        elif self.similarity_fc == 'ff':
            # 效果不是很理想
            h1 = self.w_1(input_embeddings) # B * L * H or B * 1 * H
            h2 = self.w_2(label_embeddings) # O * H or B * O * H
            h = h1.unsqueeze(-1) + h2.transpose(-1, -2).unsqueeze(-3) # B * L * H * 1 + B * 1 * H * O or 1 * H * O => B * L * H * O
            h = h.transpose(-1, -2)
            h = F.tanh(h)
            scores = self.w_3(h).squeeze(-1)
        if squeeze:
            scores = scores.squeeze(1) # B * 1 * O => B * O
        return scores

class SequenceTaggingFNNDecoder_withLabelEmbedding(nn.Module):

    def __init__(self, config, encoder_output_dim, label_embedding_dim, bilinear=False, bias=True):
        super(SequenceTaggingFNNDecoder_withLabelEmbedding, self).__init__()
        if bilinear:
            self.hidden2emb = nn.Linear(encoder_output_dim, label_embedding_dim, bias=bias)
            #init_transitions_param = torch.rand(encoder_output_dim)
            #self.hidden2emb = nn.Parameter(init_transitions_param)
        else:
            assert encoder_output_dim == label_embedding_dim
            self.hidden2emb = nn.Identity()
        self.label_embedding_dim = label_embedding_dim
        
        self.compute_similarity = SimilarityNet(config, label_embedding_dim)

    def forward(self, encoder_hiddens, label_embeddings, masked_output=None):
        """
        encoder_hiddens : B * L * H
        label_embeddings : O * H'
        """
        hidden_emb = self.hidden2emb(encoder_hiddens)
        #hidden_emb = encoder_hiddens * self.hidden2emb
        tag_logits = self.compute_similarity(hidden_emb, label_embeddings)
        if masked_output is not None:
            tag_logits = tag_logits.index_fill_(-1, masked_output, MIN_NUMBER)

        return tag_logits

#class SequenceTaggingRNNDecoder(nn.Module):
#    """encoder-decoder, seq2seq: focus && attention"""
#    def __init__(self, config, encoder_output_dim, query_vector_dim, num_labels, device=None):
#        super(SequenceTaggingRNNDecoder, self).__init__()
#class SequenceTaggingCRFDecoder(nn.Module):
#    """Conditional Random Fields"""
#    def __init__(self, config, encoder_output_dim, num_labels, device=None):
#        super(SequenceTaggingCRFDecoder, self).__init__()

#class SequenceEmbedding_TwoTails(nn.Module):
class SequenceEmbedding_Pooling(nn.Module):
    def __init__(self, config, encoder_output_dim, pooling='mean'):
        super().__init__()
        self.pooling = pooling
        assert self.pooling in ('max', 'mean')

    def forward(self, rnn_out, pooling_mask):
        """
        rnn_out : bsize x seqlen x hsize
        pooling_mask : bsize x seqlen
        """
        if self.pooling == 'mean':
            lengths = pooling_mask.sum(1).squeeze(1)
            rnn_out_pool = rnn_out.sum(1) / lengths
        elif self.pooling == 'max':
            extended_pooling_mask = pooling_mask.unsqueeze(2)
            extended_pooling_mask = (1.0 - extended_pooling_mask) * MIN_NUMBER
            masked_rnn_out = rnn_out + extended_pooling_mask
            rnn_out_pool = masked_rnn_out.max(1)[0]

        return rnn_out_pool

class SequenceEmbedding_CNN(nn.Module):
    """Hidden CNN"""
    def __init__(self, config, encoder_output_dim, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.dropout_layer = nn.Dropout(config.arch_dropout)
        self.batchnorm = nn.BatchNorm1d(encoder_output_dim)
        self.cnn = nn.Sequential(
                nn.Conv1d(encoder_output_dim, encoder_output_dim, self.kernel_size, padding=1), #, padding=self.kernel_size//2),
                nn.ReLU(),
                #nn.Dropout(p=self.dropout),
                #nn.Conv1d(encoder_output_dim, encoder_output_dim, 3, padding=self.kernel_size//2),
                #nn.ReLU(),
                )

    def forward(self, rnn_out):
        """
        rnn_out : bsize x seqlen x hsize
        """
        hiddens = rnn_out.transpose(1, 2)
        conv_hiddens = self.cnn(hiddens)
        conv_hiddens_pool = conv_hiddens.max(2)[0]
        conv_hiddens_pool = self.batchnorm(conv_hiddens_pool)
        conv_hiddens_pool = self.dropout_layer(conv_hiddens_pool)
        return conv_hiddens_pool

class SequenceEmbedding_Attention(nn.Module):
    """Hidden Attention"""
    def __init__(self, config, encoder_output_dim, query_vector_dim):
        super().__init__()
        self.dropout_layer = nn.Dropout(config.arch_dropout)
        self.Wa = nn.Linear(query_vector_dim, query_vector_dim, bias=False)
        self.Ua = nn.Conv1d(encoder_output_dim, query_vector_dim, 1, bias=False)
        self.Va = nn.Conv1d(query_vector_dim, 1, 1, bias=False)

    def forward(self, rnn_out, reversed_top_h_t, attention_mask):
        '''
        rnn_out : bsize x seqlen x hsize
        reversed_top_h_t : bsize x hsize/2
        attention_mask : bsize x seqlen
        '''
        hiddens = rnn_out.transpose(1, 2)
        reversed_top_h_t = reversed_top_h_t.squeeze(0)
        c1 = self.Wa(reversed_top_h_t)
        c2 = self.Ua(hiddens)

        c3 = c1.unsqueeze(2).repeat(1, 1, rnn_out.size(1))
        c4 = torch.tanh(c3 + c2)

        e = self.Va(c4).squeeze(1)
        mask = (1 - attention_mask).to(dtype=torch.bool)
        e.masked_fill_(mask, MIN_NUMBER)
        a = F.softmax(e, dim=1)

        context_hidden = torch.bmm(hiddens, a.unsqueeze(2)).squeeze(2)
        context_hidden = self.dropout_layer(context_hidden)
        return context_hidden
    
class SequenceClassifier_withLabelEmbedding(nn.Module):
    def __init__(self, config, encoder_output_dim, label_embedding_dim, bilinear=False, bias=True):
        super(SequenceClassifier_withLabelEmbedding, self).__init__()
        if bilinear:
            self.hidden2emb = nn.Linear(encoder_output_dim, label_embedding_dim, bias=bias)
        else:
            assert encoder_output_dim == label_embedding_dim
            self.hidden2emb = nn.Identity()
        self.label_embedding_dim = label_embedding_dim
        
        self.compute_similarity = SimilarityNet(config, label_embedding_dim)

    def forward(self, encoder_hiddens, label_embeddings, masked_output=None):
        hidden_emb = self.hidden2emb(encoder_hiddens)
        class_logits = self.compute_similarity(hidden_emb, label_embeddings)
        if masked_output is not None:
            class_logits = class_logits.index_fill_(1, masked_output, MIN_NUMBER)
        return class_logits

class SequenceClassifier_TwoTails_withLabelEmbedding(SequenceClassifier_withLabelEmbedding):

    def __init__(self, config, encoder_output_dim, label_embedding_dim, bilinear=False, bias=True):
        super().__init__(config, encoder_output_dim, label_embedding_dim, bilinear=bilinear, bias=bias)

class SequenceClassifier_Pooling_withLabelEmbedding(SequenceClassifier_withLabelEmbedding):

    def __init__(self, config, encoder_output_dim, label_embedding_dim, bilinear=False, bias=True, pooling='mean'):
        super().__init__(config, encoder_output_dim, label_embedding_dim, bilinear=bilinear, bias=bias)
        self.sequence_embeddings = SequenceEmbedding_Pooling(config, encoder_output_dim, pooling=pooling)

    def forward(self, rnn_out, pooling_mask, label_embeddings, masked_output=None):
        """
        rnn_out : bsize x seqlen x hsize
        pooling_mask : bsize x seqlen
        """
        rnn_out_pool = self.sequence_embeddings(rnn_out, pooling_mask)
        hidden_emb = self.hidden2emb(rnn_out_pool)
        class_logits = self.compute_similarity(hidden_emb, label_embeddings)
        if masked_output is not None:
            class_logits = class_logits.index_fill_(1, masked_output, MIN_NUMBER)
        return class_logits

class SequenceClassifier_CNN_withLabelEmbedding(SequenceClassifier_withLabelEmbedding):
    """Hidden CNN"""
    def __init__(self, config, encoder_output_dim, label_embedding_dim, bilinear=False, bias=True, kernel_size=3):
        super().__init__(config, encoder_output_dim, label_embedding_dim, bilinear=bilinear, bias=bias)
        self.sequence_embeddings = SequenceEmbedding_CNN(config, encoder_output_dim, kernel_size=kernel_size)

    def forward(self, rnn_out, label_embeddings, masked_output=None):
        """
        rnn_out : bsize x seqlen x hsize
        """
        conv_hiddens_pool = self.sequence_embeddings(rnn_out)
        hidden_emb = self.hidden2emb(conv_hiddens_pool)
        class_logits = self.compute_similarity(hidden_emb, label_embeddings)
        if masked_output is not None:
            class_logits = class_logits.index_fill_(1, masked_output, MIN_NUMBER)
        return class_logits

class SequenceClassifier_Attention_withLabelEmbedding(SequenceClassifier_withLabelEmbedding):
    """Hidden Attention"""
    def __init__(self, config, encoder_output_dim, query_vector_dim, label_embedding_dim, bilinear=False, bias=True):
        super().__init__(config, encoder_output_dim, label_embedding_dim, bilinear=bilinear, bias=bias)
        self.sequence_embeddings = SequenceEmbedding_Attention(config, encoder_output_dim, query_vector_dim)

    def forward(self, rnn_out, reversed_top_h_t, attention_mask, label_embeddings, masked_output=None):
        '''
        rnn_out : bsize x seqlen x hsize
        reversed_top_h_t : bsize x hsize/2
        attention_mask : bsize x seqlen
        '''
        context_hidden = self.sequence_embeddings(rnn_out, reversed_top_h_t, attention_mask)
        hidden_emb = self.hidden2emb(context_hidden)
        class_logits = self.compute_similarity(hidden_emb, label_embeddings)
        if masked_output is not None:
            class_logits = class_logits.index_fill_(1, masked_output, MIN_NUMBER)
        return class_logits
    
