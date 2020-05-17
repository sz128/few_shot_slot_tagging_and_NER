#!/usr/bin/env python3

'''
@Time   : 2018-11-29 22:11:22
@Author : su.zhu
@Desc   : 
'''

import sys
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--bio_files', type=argparse.FileType('r'), nargs='+')
parser.add_argument('--slot_vocab', required=True, help='slot vocab')
parser.add_argument('--intent_vocab', required=True, help='intent_vocab')
parser.add_argument('--multi_task_vocab_sharing_type', required=True, help='full | just_O | no')
opt = parser.parse_args()

slot_vocab_file = open(opt.slot_vocab, 'w')
intent_vocab_file = open(opt.intent_vocab, 'w')
assert opt.multi_task_vocab_sharing_type in {'full', 'just_O', 'no'}

slot_vocab = {}
intent_vocab = {}
for in_file in opt.bio_files:
    data = json.load(in_file)
    for domain_name in data:
        for batch in data[domain_name]:
            #print(list(batch['support'])) # ['seq_ins', 'labels', 'seq_outs', 'word_piece_marks', 'tokenized_texts', 'word_piece_labels']
            #print(list(batch['batch'])) # ['seq_ins', 'labels', 'seq_outs', 'word_piece_marks', 'tokenized_texts', 'word_piece_labels']
            for data_type in ['support', 'batch']:
                for slots in batch[data_type]['seq_outs']:
                    for slot in slots:
                        if opt.multi_task_vocab_sharing_type == 'full':
                            pass
                        elif opt.multi_task_vocab_sharing_type == 'just_O':
                            if slot != 'O':
                                slot = slot + '__' + domain_name
                        else:
                            slot = slot + '__' + domain_name
                        slot_vocab[slot] = 1
                for intent in batch[data_type]['labels']:
                    if opt.multi_task_vocab_sharing_type == 'full':
                        pass
                    else:
                        intent = intent + '__' + domain_name
                    intent_vocab[intent] = 1

for slot in sorted(list(slot_vocab)):
    slot_vocab_file.write(slot + '\n')
for intent in sorted(list(intent_vocab)):
    intent_vocab_file.write(intent + '\n')
