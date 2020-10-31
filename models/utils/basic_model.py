#!/usr/bin/env python3

'''
@Time   : 2019-08-08 22:47:47
@Author : su.zhu
@Desc   : 
'''

import torch
import torch.nn as nn

class BasicModel(nn.Module):

    def __init__(self):
        super(BasicModel, self).__init__()

    def load_model(self, load_dir):
        if self.device.type == 'cuda':
            self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location='cuda:0'))
        else:
            self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))

    def init_weights(self):
        """ Initialize and prunes weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        '''
        nn.Parameter will not be included.
        '''
        if hasattr(module, 'sz128__no_init_flag') and module.sz128__no_init_flag:
            #print("Module skips the initialization:", type(module), module.__class__.__name__)
            return 1
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Conv1d)):
            module.weight.data.uniform_(-0.2, 0.2)
            #module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.GRU) or isinstance(module, nn.LSTM) or isinstance(module, nn.RNN):
            for name, param in module.named_parameters():
                if 'weight_ih' in name or 'weight_hh' in name:
                    param.data.uniform_(-0.2, 0.2)
                    #param.data.normal_(mean=0.0, std=0.02)
                elif 'bias' in name:
                    param.data.zero_()
        else:
            print("Module skips the initialization:", type(module))
        if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
            module.bias.data.zero_()

