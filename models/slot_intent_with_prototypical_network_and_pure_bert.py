#!/usr/bin/env python3

'''
@Time   : 2019-08-08 18:49:42
@Author : su.zhu
@Desc   : 
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCELoss #, NLLLoss

from models.utils.basic_model import BasicModel

import models.utils.basic_modules as basic_modules_with_label_embedding
## adaptive CRF
from models.utils.crf import GeneralTransitionLayer, AbstractTransitionLayer, TransitionLayer, CRFLoss

#from utils.pretrained_transformer import transformer_forward_by_ignoring_suffix
from models.predictor import Projection, MatchingClassifier

MIN_NUMBER = -1e32 # -float('inf')
EPS = 1e-16

def get_unit_vectors(x, eps=EPS):
    """ x : B * L * H or B * H """
    x_l2_norm = torch.norm(x, p=2, dim=-1, keepdim=True) + eps
    x = x.div(x_l2_norm.expand_as(x))
    return x

def transformer_forward_by_ignoring_suffix(pretrained_top_hiddens, batch_size, max_word_length, selects, copies, device=None):
    '''
    Ignore hidden states of all suffixes: [CLS] from ... to de ##n ##ver [SEP] => from ... to de
    '''
    batch_size, pretrained_seq_length, hidden_size = pretrained_top_hiddens.size(0), pretrained_top_hiddens.size(1), pretrained_top_hiddens.size(2)
    #chosen_encoder_hiddens = pretrained_top_hiddens.view(-1, hidden_size).index_select(0, selects)
    chosen_encoder_hiddens = pretrained_top_hiddens.reshape(-1, hidden_size).index_select(0, selects)
    embeds = torch.zeros(batch_size * max_word_length, hidden_size, device=device)
    embeds = embeds.index_copy_(0, copies, chosen_encoder_hiddens).view(batch_size, max_word_length, -1)
    return embeds

class SequenceEncoder_with_pure_bert(BasicModel):

    def __init__(self, config, pretrained_tf_model, device=None):
        super().__init__()
        
        self.tf_model = pretrained_tf_model
        
        #self.embeddings = self.tf_model.embeddings # NOTE: reference of nn.module is not appropriate for reproducing
        #self.encoder = self.tf_model.encoder
        #self.pooler = self.tf_model.pooler

        self.device = device

        self.dropout_layer = nn.Dropout(config.arch_dropout)

    def get_output_dim(self):
        return self.tf_model.config.hidden_size
	
	#def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
    def get_bert_output_in_encoder_part(self, embedding_output, attention_mask):
        input_shape = embedding_output.size()[:-1]
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                    "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                        input_shape, attention_mask.shape
                    )
            )
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e16 #-10000.0

        head_mask = [None] * self.tf_model.config.num_hidden_layers

        encoder_outputs = self.tf_model.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
        )
        sequence_output = encoder_outputs[0]
        if hasattr(self.tf_model, 'pooler'): # bert
            pooled_output = sequence_output[:, 0] #self.tf_model.pooler(sequence_output)
        else: # electra
            pooled_output = sequence_output[:, 0]

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

    def forward_ori_bert(self, input_ids, token_type_ids, attention_mask):
        outputs = self.tf_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        sequence_output, pooled_output = outputs[:2]
        return sequence_output, pooled_output

    def forward(self, inputs, slot_embeddings=None, lattice=None):
        #slot_embeddings: O * H
        input_ids = inputs["input_tf"]["input_ids"]
        position_ids = None # default
        token_type_ids = inputs["input_tf"]["segment_ids"]
        attention_mask = inputs["input_tf"]["attention_mask"]
        batch_size = inputs["input_tf"]["batch_size"]
        max_word_length = inputs["input_tf"]["max_word_length"]
        selects = inputs["input_tf"]["selects"]
        copies = inputs["input_tf"]["copies"]
        if slot_embeddings is None:
            token_embeds = self.tf_model.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
            bert_outputs = self.get_bert_output_in_encoder_part(token_embeds, attention_mask)
            bert_top_hiddens, bert_pooled_output = bert_outputs[0:2]
        else:
            max_token_length = input_ids.size()[1]
            slot_number = slot_embeddings.size()[0]
            token_embeds = self.tf_model.embeddings(input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
            extended_slot_embeddings = slot_embeddings[None, :, :] #.repeat(batch_size, 1, 1)
            slot_nodes_position_ids = torch.zeros(batch_size, slot_number, dtype=torch.long, device=self.device) # pos all 0
            slot_nodes_type_ids = torch.ones(batch_size, slot_number, dtype=torch.long, device=self.device) # seg id = 1
            extended_slot_embeds = self.tf_model.embeddings(input_ids=None, position_ids=slot_nodes_position_ids, token_type_ids=slot_nodes_type_ids, inputs_embeds=extended_slot_embeddings)
            combined_input_embeds = torch.cat((token_embeds, extended_slot_embeds), dim=1)

            slot_attention_mask = torch.ones(batch_size, slot_number, dtype=attention_mask.dtype, device=self.device)
            combined_attention_mask = torch.cat((attention_mask, slot_attention_mask), dim=1)

            combined_bert_outputs = self.get_bert_output_in_encoder_part(combined_input_embeds, combined_attention_mask)
            combined_bert_top_hiddens, bert_pooled_output = combined_bert_outputs[0:2]
            bert_top_hiddens = combined_bert_top_hiddens[:, :max_token_length, :] #.contiguous()
            query_aware_slot_embeddings = combined_bert_top_hiddens[:, max_token_length:, :]
        word_embeds = transformer_forward_by_ignoring_suffix(bert_top_hiddens, batch_size, max_word_length, selects, copies, device=self.device)
        word_embeds, bert_pooled_output = self.dropout_layer(word_embeds), self.dropout_layer(bert_pooled_output)

        if slot_embeddings is None:
            return word_embeds, bert_pooled_output
        else:
            return word_embeds, bert_pooled_output, query_aware_slot_embeddings

class FewShotIntentSlot_ProtoNet(BasicModel):

    def __init__(self, config, special_slot_label_number, slot_embedding_type='with_BIO', matching_similarity_type='xy1', sentence_encoder_shared=True, slot_embedding_dependent_query_encoding='none', matching_similarity_y='ctx', matching_similarity_function='dot', task_adaptive_projection='none', pretrained_tf_model=None, device=None):
        '''
        slot_embedding_type: with_BIO, with_BI, without_BIO
        matching_similarity_type: xy, x1y, xy1, x1y1, rxy
        slot_embedding_dependent_query_encoding: none, ctx, desc
        matching_similarity_y: ctx, desc, ctx_desc
        task_adaptive_projection: none, LSM, adaptive_finetune, LEN
        '''
        super().__init__()

        self.config = config
        self.device = device
        self.special_slot_label_number = special_slot_label_number
        self.slot_embedding_type = slot_embedding_type
        self.matching_similarity_type = matching_similarity_type
        self.matching_similarity_y = matching_similarity_y

        #sentence_encoder_shared = True #default 'True'; False
        bilinear = False #default 'False'; True, False
        crf_trainable_balance_weight = False # default 'False'; False
        if self.matching_similarity_type == 'rxy':
            crf_trainable_balance_weight = True
        
        #slot_embedding_dependent_query_encoding = True # True, False
        self.slot_embedding_dependent_query_encoding = slot_embedding_dependent_query_encoding

        # token embedding layer (shared between query and support sets)
        self.query_sentence_encoder = SequenceEncoder_with_pure_bert(config, pretrained_tf_model, device=self.device)
        if sentence_encoder_shared:
            self.support_sentence_encoder = self.query_sentence_encoder
        else:
            self.support_sentence_encoder = SequenceEncoder_with_pure_bert(config, pretrained_tf_model, device=self.device)
        encoder_output_dim = self.query_sentence_encoder.get_output_dim()
        self.special_slot_label_embeddings = nn.Embedding(special_slot_label_number, encoder_output_dim)

        self.slot_tagger_projection_x = Projection(config, encoder_output_dim, device=self.device)
        self.slot_tagger_projection_y = self.slot_tagger_projection_x
        self.slot_tagger = MatchingClassifier(config, matching_similarity_function=matching_similarity_function)
        projected_tag_embedding_dim = self.slot_tagger_projection_y.get_output_dim()
        if config.task_st == "slot_tagger":
            self.slot_tag_loss_fct = CrossEntropyLoss(ignore_index=config.output_tag_pad_id, size_average=False)
        elif config.task_st == "slot_tagger_with_adaptive_crf":
            self.crf_transition_layer = TransitionLayer(projected_tag_embedding_dim)
            self.crf = CRFLoss(trainable_balance_weight=crf_trainable_balance_weight, device=self.device)
            self.slot_tag_loss_fct = self.crf.neg_log_likelihood_loss
        elif config.task_st == "slot_tagger_with_abstract_crf":
            self.crf_transition_layer = AbstractTransitionLayer(device=self.device)
            self.crf = CRFLoss(trainable_balance_weight=crf_trainable_balance_weight, device=self.device)
            self.slot_tag_loss_fct = self.crf.neg_log_likelihood_loss
        else:
            exit()
        
        self.intent_multi_class = (config.task_sc_type == "multi_cls_BCE")
        self.intent_classifier = MatchingClassifier(config, matching_similarity_function=matching_similarity_function)
        if self.intent_multi_class:
            self.intent_loss_fct = BCELoss(size_average=False)
        else:
            self.intent_loss_fct = CrossEntropyLoss(size_average=False)

        self.init_weights()
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if hasattr(module, 'sz128__no_init_flag') and module.sz128__no_init_flag:
            #print("Module skips the initialization:", type(module), module.__class__.__name__)
            return 1
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.GRU) or isinstance(module, nn.LSTM) or isinstance(module, nn.RNN):
            for name, param in module.named_parameters():
                if 'weight_ih' in name or 'weight_hh' in name:
                    #param.data.uniform_(-0.2, 0.2)
                    param.data.normal_(mean=0.0, std=0.02)
                elif 'bias' in name:
                    param.data.zero_()
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_contextual_label_embeddings_from_support_set(self, inputs, lengths,
            sentence_mask, slot_label_indicator, intent_label_indicator,
            slot_head_labels_token_ids, slot_other_labels_seg_ids=None,
            slot_other_labels_selected_slot_ids=None, addtional_features=None,
            lattice=None):
        tf_out, snt_vector = self.support_sentence_encoder(inputs, lattice=lattice)	
        batch_size, max_length, hidden_size = tf_out.size()

        if self.slot_embedding_type == 'with_BIO' or self.slot_embedding_type == 'with_BI':
            other_slot_tag_embeds = torch.matmul(slot_label_indicator, tf_out.reshape((batch_size * max_length, hidden_size)))
        else:
            other_slot_embeds = torch.matmul(slot_label_indicator, tf_out.reshape((batch_size * max_length, hidden_size)))
            other_slot_tag_name_embeds = torch.index_select(other_slot_embeds, 0, slot_other_labels_selected_slot_ids)
            other_slot_tag_seg_embeds = self.special_slot_label_embeddings(slot_other_labels_seg_ids)
            other_slot_tag_embeds = other_slot_tag_seg_embeds + other_slot_tag_name_embeds
        head_slot_tag_embeds = self.special_slot_label_embeddings(slot_head_labels_token_ids)
        slot_tag_embeds = torch.cat((head_slot_tag_embeds, other_slot_tag_embeds), dim=0)

        intent_embeds = torch.matmul(intent_label_indicator, snt_vector)
        
        return (other_slot_tag_embeds, slot_tag_embeds), intent_embeds

    def get_description_based_label_embeddings_from_support_set(self, input_ids, token_type_ids, attention_mask, slot_head_labels_token_ids):
        tf_out, snt_vector = self.support_sentence_encoder.forward_ori_bert(input_ids, token_type_ids, attention_mask)	
        other_slot_tag_embeds = snt_vector
        head_slot_tag_embeds = self.special_slot_label_embeddings(slot_head_labels_token_ids)
        slot_tag_embeds = torch.cat((head_slot_tag_embeds, other_slot_tag_embeds), dim=0)
        
        return (other_slot_tag_embeds, slot_tag_embeds)

    def get_label_embeddings_from_support_set(self, inputs, lengths,
            sentence_mask, slot_label_indicator, intent_label_indicator,
            slot_head_labels_token_ids, slot_other_labels_seg_ids=None,
            slot_other_labels_selected_slot_ids=None, addtional_features=None,
            label_desc_inputs=None, lattice=None):
        if label_desc_inputs is None:
            assert self.matching_similarity_y == 'ctx'
            desc_other_slot_tag_embeds, desc_slot_tag_embeds = None, None
        else:
            desc_other_slot_tag_embeds, desc_slot_tag_embeds = self.get_description_based_label_embeddings_from_support_set(**label_desc_inputs, slot_head_labels_token_ids=slot_head_labels_token_ids)
            H = desc_slot_tag_embeds.shape[-1]
        if self.matching_similarity_y != 'desc':
            (ctx_other_slot_tag_embeds, ctx_slot_tag_embeds), intent_embeds = self.get_contextual_label_embeddings_from_support_set(inputs,
                    lengths, sentence_mask, slot_label_indicator,
                    intent_label_indicator, slot_head_labels_token_ids,
                    slot_other_labels_seg_ids=slot_other_labels_seg_ids,
                    slot_other_labels_selected_slot_ids=slot_other_labels_selected_slot_ids,
                    addtional_features=addtional_features, lattice=lattice)
        else:
            intent_embeds = torch.ones(intent_label_indicator.shape[0], H, device=self.device) # not ready for intent

        ## maybe you should build slot_tagger_projection first
        ## ...
        if self.matching_similarity_y != 'desc':
            ctx_slot_tag_embeds = self.slot_tagger_projection_y(ctx_slot_tag_embeds) # M(l)
        if not (label_desc_inputs is None):
            desc_slot_tag_embeds = self.slot_tagger_projection_y(desc_slot_tag_embeds) # M(l)
        if self.matching_similarity_type in {'xy1', 'x1y1'}:
            if self.matching_similarity_y != 'desc':
                ctx_slot_tag_embeds = get_unit_vectors(ctx_slot_tag_embeds, eps=0)
            intent_embeds = get_unit_vectors(intent_embeds, eps=0)
            if not (label_desc_inputs is None):
                desc_slot_tag_embeds = get_unit_vectors(desc_slot_tag_embeds, eps=0)
        
        if self.matching_similarity_y == 'ctx':
            slot_tag_embeds = ctx_slot_tag_embeds
        elif self.matching_similarity_y == 'desc':
            slot_tag_embeds = desc_slot_tag_embeds
        elif self.matching_similarity_y == 'ctx_desc':
            lambda_ = 0.9
            slot_tag_embeds = lambda_ * ctx_slot_tag_embeds + (1 - lambda_) * desc_slot_tag_embeds

        if self.slot_embedding_dependent_query_encoding == 'ctx':
            other_slot_tag_embeds = ctx_other_slot_tag_embeds
        elif self.slot_embedding_dependent_query_encoding == 'desc':
            other_slot_tag_embeds = desc_other_slot_tag_embeds
        elif self.slot_embedding_dependent_query_encoding == 'none':
            other_slot_tag_embeds = None
        
        return (other_slot_tag_embeds, slot_tag_embeds), intent_embeds

    def get_feature_representations_for_query_set(self, slot_tag_embeds,
            intent_embeds, inputs, lattice=None):
        other_slot_tag_embeds, slot_tag_embeds = slot_tag_embeds
        if self.slot_embedding_dependent_query_encoding != 'none':
            tf_out, snt_vector, query_aware_slot_tag_embeds = self.query_sentence_encoder(inputs, slot_embeddings=other_slot_tag_embeds, lattice=lattice)
            #slot_tag_embeds = query_aware_slot_tag_embeds # NOTE
        else:
            tf_out, snt_vector = self.query_sentence_encoder(inputs, lattice=lattice)
        
        return tf_out

    def forward(self, slot_tag_embeds, intent_embeds, inputs, lengths, sentence_mask,
            addtional_features=None, slot_tag_masked_output=None,
            intent_masked_output=None, slot_tags=None, intents=None,
            slot_tag_to_id=None, detach=False, lattice=None):

        other_slot_tag_embeds, slot_tag_embeds = slot_tag_embeds
        if detach:
            other_slot_tag_embeds, slot_tag_embeds = other_slot_tag_embeds.detach() if other_slot_tag_embeds is not None else None, slot_tag_embeds.detach()
        if self.slot_embedding_dependent_query_encoding != 'none':
            tf_out, snt_vector, query_aware_slot_tag_embeds = self.query_sentence_encoder(inputs, slot_embeddings=other_slot_tag_embeds, lattice=lattice)
            #slot_tag_embeds = query_aware_slot_tag_embeds # NOTE
        else:
            tf_out, snt_vector = self.query_sentence_encoder(inputs, lattice=lattice)
        if detach:
            tf_out, snt_vector = tf_out.detach(), snt_vector.detach()

        tf_out = self.slot_tagger_projection_x(tf_out) ## M(f(x))
        if self.matching_similarity_type in {'x1y', 'x1y1'}:
            tf_out, snt_vector = get_unit_vectors(tf_out), get_unit_vectors(snt_vector)

        slot_tag_logits = self.slot_tagger(tf_out, slot_tag_embeds, masked_output=slot_tag_masked_output)
        intent_logits = self.intent_classifier(snt_vector, intent_embeds, masked_output=intent_masked_output)
        
        outputs = (slot_tag_logits, intent_logits, tf_out, snt_vector)
        if slot_tags is not None and intents is not None:
            if self.config.task_st == "slot_tagger":
                slot_tag_loss = self.slot_tag_loss_fct(slot_tag_logits.transpose(1,2), slot_tags) # B * C * L, B * L
            elif self.config.task_st == "slot_tagger_with_adaptive_crf":
                if self.matching_similarity_type in {'xy1', 'x1y1'}:
                    slot_tag_embeds_tmp = slot_tag_embeds
                else:
                    slot_tag_embeds_tmp = get_unit_vectors(slot_tag_embeds, eps=0)
                transitions = self.crf_transition_layer(slot_tag_embeds_tmp)
                slot_tag_loss = self.slot_tag_loss_fct(transitions, slot_tag_logits, sentence_mask.bool(), slot_tags)
            elif self.config.task_st == "slot_tagger_with_abstract_crf":
                transitions = self.crf_transition_layer(slot_tag_to_id)
                slot_tag_loss = self.slot_tag_loss_fct(transitions, slot_tag_logits, sentence_mask.bool(), slot_tags)
            else:
                pass

            if self.intent_multi_class:
                intent_loss = self.intent_loss_fct(torch.sigmoid(intent_logits), intents)
            else:
                intent_loss = self.intent_loss_fct(intent_logits, intents)

            outputs = (slot_tag_loss, intent_loss) + outputs
        
        return outputs

    def decode_top_hyp(self, slot_tag_embeds, slot_tag_logits, intent_logits, sentence_mask, slot_tag_to_id=None):
        other_slot_tag_embeds, slot_tag_embeds = slot_tag_embeds # for slot_tagger_with_adaptive_crf
        if self.config.task_st == "slot_tagger":
            slot_tag_top_hyp = slot_tag_logits.data.cpu().numpy().argmax(axis=-1)
        elif self.config.task_st == "slot_tagger_with_adaptive_crf":
            if self.matching_similarity_type in {'xy1', 'x1y1'}:
                slot_tag_embeds_tmp = slot_tag_embeds
            else:
                slot_tag_embeds_tmp = get_unit_vectors(slot_tag_embeds, eps=0)
            transitions = self.crf_transition_layer(slot_tag_embeds_tmp)
            tag_path_scores_top_hyp, tag_path_top_hyp = self.crf.viterbi_decode(transitions, slot_tag_logits, sentence_mask.bool())
            slot_tag_top_hyp = tag_path_top_hyp.data.cpu().numpy()
        elif self.config.task_st == "slot_tagger_with_abstract_crf":
            transitions = self.crf_transition_layer(slot_tag_to_id)
            tag_path_scores_top_hyp, tag_path_top_hyp = self.crf.viterbi_decode(transitions, slot_tag_logits, sentence_mask.bool())
            slot_tag_top_hyp = tag_path_top_hyp.data.cpu().numpy()
        else:
            pass

        if self.intent_multi_class:
            intent_top_hyp = torch.sigmoid(intent_logits).data.cpu().numpy()
        else:
            intent_top_hyp = intent_logits.data.cpu().numpy().argmax(axis=-1)

        return slot_tag_top_hyp, intent_top_hyp

    def decode_by_similarity_of_BERT(self, support_inputs,
            support_sentence_mask, support_slot_tags, support_intents,
            query_inputs, pred_intent=False, support_lattice=None, query_lattice=None):
        s_tf_out, s_snt_vector = self.support_sentence_encoder(support_inputs, lattice=support_lattice)
        s_tf_out, s_snt_vector = get_unit_vectors(s_tf_out), get_unit_vectors(s_snt_vector)
        s_batch_size, s_length = s_tf_out.size(0), s_tf_out.size(1)
        tf_out, snt_vector = self.query_sentence_encoder(query_inputs, lattice=query_lattice)
        q_batch_size, q_length = tf_out.size(0), tf_out.size(1)
        
        output_mask = ((1.0 - support_sentence_mask.view(-1)) * (-1e16))[None, None, :]
        slot_tag_logits = torch.matmul(tf_out, s_tf_out.contiguous().view(s_batch_size * s_length, -1).transpose(0, 1)) + output_mask
        slot_tag_top_hyp_ids = torch.argmax(slot_tag_logits, dim=-1)
        slot_tag_top_hyp = torch.index_select(support_slot_tags.view(-1), 0, slot_tag_top_hyp_ids.view(-1)).view(q_batch_size, q_length)
        slot_tag_top_hyp = slot_tag_top_hyp.data.cpu().numpy()
        
        if not pred_intent:
            return slot_tag_top_hyp
        else:
            intent_logits = torch.matmul(snt_vector, s_snt_vector.transpose(0, 1))
            intent_top_hyp_ids = torch.argmax(intent_logits, dim=-1)
            intent_top_hyp = torch.index_select(support_intents.view(-1), 0, intent_top_hyp_ids.view(-1))
            intent_top_hyp = intent_top_hyp.data.cpu().numpy()
            return slot_tag_top_hyp, intent_top_hyp
    
    def decompress_CRF_transitions(self, slot_tag_to_id):
        assert self.config.task_st == "slot_tagger_with_abstract_crf"
        transitions = self.crf_transition_layer(slot_tag_to_id)
        self.crf_transition_layer = GeneralTransitionLayer(transitions)
    
    def compress_CRF_transitions(self):
        self.crf_transition_layer = AbstractTransitionLayer(device=self.device)
        self.crf_transition_layer.to(self.device)

