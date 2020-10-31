import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class GeneralTransitionLayer(nn.Module):
    def __init__(self, transitions):
        super().__init__()
        
        self.transitions = nn.Parameter(transitions.data.clone())

    def forward(self, *args):
        return self.transitions

class AbstractTransitionLayer(nn.Module):
    ''' https://arxiv.org/abs/1906.08711 '''
    '''
        O   sB  dB  sI  dI
    O   0.6 0.4 /   0   /
    B   0.3 0.1 0.1 0.5 0
    I   0.6 0.1 0.1 0.2 0
    '''
    def __init__(self, device=None):
        super().__init__()

        init_transitions_param = torch.zeros(3, 5)
        self.crf_transitions_model = nn.Parameter(init_transitions_param)
        #nn.init.uniform_(self.crf_transitions_model, -0.01, 0.01)

        self.device = device
        self.ids_map = {
                'O': {'O': 0, 'sB': 1, 'dB': 2, 'sI': 3, 'dI': 4},
                'B': {'O': 5, 'sB': 6, 'dB': 7, 'sI': 8, 'dI': 9},
                'I': {'O': 10, 'sB': 11, 'dB': 12, 'sI': 13, 'dI': 14}
                }
        self.last_memory_label_to_id = None
        self.last_memory_selected_ids = None

    def forward(self, label_to_id):
        selected_ids = self.get_selected_ids_from_label_list(label_to_id)
        label_size = selected_ids.size(0)
        transitions_matrix = self.crf_transitions_model.view(-1).index_select(0, selected_ids.view(-1))
        transitions_matrix = transitions_matrix.view(label_size, label_size)
        return transitions_matrix

    def get_selected_ids_from_label_list(self, label_to_id):
        if self.last_memory_label_to_id == label_to_id:
            return self.last_memory_selected_ids
        else:
            label_size = len(label_to_id)
            labels = {}
            for label in label_to_id:
                if '-' in label:
                    bio, name = label.split('-', 1)
                elif label == 'O':
                    bio, name = 'O', 'O'
                else:
                    bio, name = label, None
                labels[label_to_id[label]] = (bio, name)
            selected_ids = torch.zeros(label_size, label_size)
            for i in range(label_size):
                bio_1, name_1 = labels[i]
                for j in range(label_size):
                    bio_2, name_2 = labels[j]
                    if name_1 == None or name_2 == None:
                        selected_ids[i][j] = self.ids_map['O']['dB'] # O->sI, invalid
                    elif name_1 == 'O' and name_2 == 'O':
                        selected_ids[i][j] = self.ids_map[bio_1][bio_2]
                    elif name_1 == 'O' and name_2 != 'O':
                        selected_ids[i][j] = self.ids_map[bio_1]['s' + bio_2]
                    elif name_1 != 'O' and name_2 == 'O':
                        selected_ids[i][j] = self.ids_map[bio_1][bio_2]
                    elif name_1 == name_2:
                        selected_ids[i][j] = self.ids_map[bio_1]['s' + bio_2]
                    else:
                        selected_ids[i][j] = self.ids_map[bio_1]['d' + bio_2]
            
            selected_ids = selected_ids.to(dtype=torch.long, device=self.device)
            self.last_memory_label_to_id = label_to_id
            self.last_memory_selected_ids = selected_ids
            return selected_ids
                    
class TransitionLayer_v1(nn.Module):
    ''' nn.Bilinear takes more memory in computation '''
    def __init__(self, label_embedding_dim):
        super().__init__()

        self.crf_transitions_model = nn.Bilinear(label_embedding_dim, label_embedding_dim, 1, bias=False)
        self.crf_transitions_model.weight.data.zero_()

    def forward(self, label_embeddings):
        """
        label_embeddings : O * H
        """
        label_set_size = label_embeddings.size(0)
        left = label_embeddings[None, :, :].expand(label_set_size, -1, -1).contiguous()
        right = label_embeddings[:, None, :].expand(-1, label_set_size, -1).contiguous()
        # # transitions (f_tag_size, t_tag_size), transition value from f_tag to t_tag
        transitions_matrix = self.crf_transitions_model(left, right).squeeze(2)
        
        return transitions_matrix


class TransitionLayer(nn.Module):
    def __init__(self, label_embedding_dim):
        super().__init__()

        init_transitions_param = torch.zeros(label_embedding_dim, label_embedding_dim)
        #init_transitions_param = torch.zeros(label_embedding_dim, 1)
        self.crf_transitions_model = nn.Parameter(init_transitions_param)
        self.label_embedding_dim = label_embedding_dim

    def forward(self, label_embeddings):
        """
        label_embeddings : O * H or B * O * H
        """
        label_set_size = label_embeddings.size(0)
        left = label_embeddings
        right = label_embeddings.transpose(0, 1) #.contiguous()
        # # transitions (f_tag_size, t_tag_size), transition value from f_tag to t_tag
        if len(label_embeddings.shape) == 2:
            transitions_matrix = torch.matmul(torch.matmul(left, self.crf_transitions_model), right) #/ math.sqrt(self.label_embedding_dim)
        else:
            transitions_matrix = torch.matmul(torch.matmul(left, self.crf_transitions_model[None, :, :]), right) #/ math.sqrt(self.label_embedding_dim)
        #transitions_matrix = torch.matmul(torch.matmul(left, self.crf_transitions_model.expand(-1, self.label_embedding_dim)), right)
        
        return transitions_matrix

class CRFLoss(nn.Module):

    def __init__(self, trainable_balance_weight=False, device=None):
        super().__init__()
        
        self.device = device

        # # transitions (f_tag_size, t_tag_size), transition value from f_tag to t_tag
        #init_transitions = torch.zeros(self.tagset_size, self.tagset_size)
        #self.transitions = nn.Parameter(init_transitions)
        if trainable_balance_weight:
            self.scaling_feats = nn.Parameter(torch.rand(1, dtype=torch.float, device=device)) #a uniform distribution on the interval [0, 1)[0,1)
        else:
            self.scaling_feats = 1

    def _calculate_alg(self, transitions, feats, mask):
        """
            input:
                feats: (batch, seq_len, self.tag_size)
                masks: (batch, seq_len)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        tagset_size = transitions.size(0)
        assert(tag_size == tagset_size)
        mask = mask.transpose(1,0).contiguous()
        assert mask[0].all() ## minimal length is >= 1

        first_input = feats[:, 0, :]
        if seq_len == 1:
            final_partition = torch.logsumexp(first_input, 1) # bat_size * to_target
        else:
            # This matrix is expanded into a [1, num_tags, num_tags] in preparation for the broadcast summation
            partition = first_input
            transitions_expand = transitions.view(1,tag_size,tag_size)
            for idx in range(1, seq_len):
                partition = partition.contiguous().view(batch_size, tag_size, 1) #.expand(batch_size, tag_size, tag_size)
                transition_scores = partition + transitions_expand 
                cur_partition = feats[:, idx, :] + torch.logsumexp(transition_scores, 1) # bat_size * to_target
                
                mask_idx = mask[idx, :].view(batch_size, 1) #.expand(batch_size, tag_size)
                masked_cur_partition = cur_partition.masked_select(mask_idx)
                ## let mask_idx broadcastable, to disable warning
                mask_idx = mask_idx.contiguous().view(batch_size, 1, 1)

                ## replace the partition where the maskvalue=1, other partition value keeps the same
                partition.masked_scatter_(mask_idx, masked_cur_partition)
            final_partition = torch.logsumexp(partition, 1)
        return final_partition.sum()

    def _viterbi_decode(self, transitions, feats, mask):
        """
            input:
                feats: (batch, seq_len, tag_size)
                mask: (batch, seq_len)
            output:
                decode_idx: (batch, seq_len) decoded sequence
                path_score: (batch, 1) corresponding score for each sequence (to be implementated)
        """
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)
        tagset_size = transitions.size(0)
        assert(tag_size == tagset_size)
        mask = mask.transpose(1,0).contiguous()
        assert mask[0].all() ## minimal length is >= 1
        
        if seq_len == 1:
            path_score, decode_idx = torch.max(feats, dim=2)
            path_score = path_score.squeeze(1)
        else:
            back_points = list()
            partition_history = list()
            mask_reverse = (1 - mask.long()).bool()

            first_input = feats[:, 0, :] # bat_size * to_target
            # This matrix is expanded into a [1, num_tags, num_tags] in preparation for the broadcast summation
            partition = first_input
            partition_history.append(partition)
            #back_points.append(partition)
            transitions_expand = transitions.view(1,tag_size,tag_size)
            for idx in range(1, seq_len):
                partition = partition.contiguous().view(batch_size, tag_size, 1) #.expand(batch_size, tag_size, tag_size)
                transition_scores = partition + transitions_expand 
                cur_scores, cur_backpointers = torch.max(transition_scores, 1) # bat_size * to_target
                partition = feats[:, idx, :] + cur_scores
                partition_history.append(partition)
                
                ## cur_bp: (batch_size, tag_size) max source score position in current tag
                ## set padded label as 0, which will be filtered in post processing
                cur_backpointers.masked_fill_(mask_reverse[idx, :].view(batch_size, 1), 0)
                back_points.append(cur_backpointers)
            partition_history = torch.cat(partition_history, dim=1).view(batch_size, seq_len, tag_size)
            length_mask = torch.sum(mask.transpose(1, 0), dim=1).view(batch_size, 1).long()
            last_position = length_mask.view(batch_size, 1, 1).expand(batch_size, 1, tag_size) - 1
            last_partition = torch.gather(partition_history, 1, last_position).view(batch_size, tag_size)
            ## best score and last tags
            path_score, pointer = torch.max(last_partition, 1)
            ## move the end ids(expand to tag_size) to the corresponding position of back_points to replace the 0 values
            insert_last = pointer.contiguous().view(batch_size, 1, 1).expand(batch_size, 1, tag_size)
            pad_zero = torch.zeros(batch_size, tag_size).to(self.device, dtype=torch.long)
            back_points.append(pad_zero)  # 考虑 len == max_len 的那些句子
            back_points = torch.cat(back_points, dim=1).view(batch_size, seq_len, tag_size)
            back_points.scatter_(1, last_position, insert_last)
            ## decode from the end, padded position ids are 0, which will be filtered if following evaluation
            decode_idx = torch.zeros(batch_size, seq_len, dtype=torch.long, device=self.device)
            decode_idx[:, -1] = pointer.data
            for idx in range(back_points.size(1) - 2, -1, -1):
                pointer = torch.gather(back_points[:, idx, :], 1, pointer.contiguous().view(batch_size, 1)).squeeze(1)
                decode_idx[:, idx] = pointer.data.view(batch_size)
        return path_score, decode_idx

    def viterbi_decode(self, transitions, feats, mask):
        feats *= self.scaling_feats
        path_score, best_path = self._viterbi_decode(transitions, feats, mask)
        return path_score, best_path

    def _score_sentence(self, transitions, feats, mask, tags):
        """
            input:
                feats: (batch, seq_len, tag_size)
                mask: (batch, seq_len)
                tags: tensor  (batch, seq_len)
            output:
                score: sum of score for gold sequences within whole batch
        """
        # Gives the score of a provided tag sequence
        batch_size = feats.size(0)
        seq_len = feats.size(1)
        tag_size = feats.size(2)

        ## crf_unary_score(tag_indices, sequence_lengths, inputs)
        unary_scores = torch.gather(feats, 2, tags.view(batch_size, seq_len, 1)).squeeze(2) # batch_size * seq_len
        unary_scores = unary_scores.masked_select(mask)

        ## crf_binary_score(tag_indices, sequence_lengths, transition_params)
        start_tag_indices = tags[:, :-1]
        end_tag_indices = tags[:, 1:]
        flattened_transition_indices = start_tag_indices * tag_size + end_tag_indices
        binary_scores = torch.gather(transitions.view(-1), 0, flattened_transition_indices.view(-1))
        binary_scores = binary_scores.masked_select(mask[:, 1:].contiguous().view(-1))

        gold_score = unary_scores.sum() + binary_scores.sum()
        return gold_score

    def neg_log_likelihood_loss(self, transitions, feats, mask, tags):
        # nonegative log likelihood
        feats *= self.scaling_feats
        forward_score = self._calculate_alg(transitions, feats, mask)
        gold_score = self._score_sentence(transitions, feats, mask, tags)
        return forward_score - gold_score

    #def _viterbi_decode_nbest(self, feats, mask, nbest):
