#!/usr/bin/env python3

'''
@Time   : 2019-08-09 16:17:14
@Author : su.zhu
@Desc   : 
'''

import collections
import torch

def read_vocab_file(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip('\n')
        vocab[token] = index
    return vocab

def save_vocab_file(token_to_id, vocab_file):
    index = 0
    with open(vocab_file, 'w') as writer:
        for token, token_index in sorted(token_to_id.items(), key=lambda kv: kv[1]):
            if index != token_index:
                print("Saving vocabulary to {}: vocabulary indices are not consecutive."
                        " Please check that the vocabulary is not corrupted!".format(vocab_file))
                index = token_index
            writer.write(token + u'\n')
            index += 1

def read_input_vocab_from_data_file(file_name, word_tokenizer=None, char_tokenizer=None, lowercase=False, mini_word_freq=1, with_tag=True, separator=':'):
    """
    data_line: I:O want:O to:O fly:O to:O Shanghai:B-to_city <=> find_flight
    """
    assert word_tokenizer is not None or char_tokenizer is not None
    print('Constructing input vocabulary from ', file_name, ' ...')
    all_tokens = {}
    with open(file_name, 'r') as f:
        for line in f:
            slot_tag_line = line.strip('\n\r').split(' <=> ')[0]
            if slot_tag_line == "":
                continue
            for item in slot_tag_line.split(' '):
                if with_tag:
                    tmp = item.split(separator)
                    assert len(tmp) >= 2
                    word, tag = separator.join(tmp[:-1]), tmp[-1]
                else:
                    word = item
                if lowercase:
                    word = word.lower()

                if word not in all_tokens:
                    all_tokens[word] = 1
                else:
                    all_tokens[word] += 1

    if word_tokenizer is not None:
        sorted_all_tokens = sorted(all_tokens.items(), key=lambda x:x[1], reverse=True)
        selected_tokens = [x[0] for x in sorted_all_tokens if x[1] >= mini_word_freq]
        for token in selected_tokens:
            if token not in word_tokenizer.token_to_id:
                word_tokenizer.token_to_id[token] = word_tokenizer.vocab_size
                word_tokenizer.vocab_size += 1
        word_tokenizer.id_to_token = collections.OrderedDict([(ids, tok) for tok, ids in word_tokenizer.token_to_id.items()])

    if char_tokenizer is not None:
        for word in all_tokens:
            for char in word:
                if char not in char_tokenizer.token_to_id:
                    char_tokenizer.token_to_id[char] = char_tokenizer.vocab_size
                    char_tokenizer.vocab_size += 1
        char_tokenizer.id_to_token = collections.OrderedDict([(ids, tok) for tok, ids in char_tokenizer.token_to_id.items()])

def whitespace_tokenize(text, lowercase=False):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = [token.lower() if lowercase else token for token in text.split()]
    return tokens

class Tokenizer():

    SPECIAL_TOKENS_ATTRIBUTES = ["bos_token", "eos_token", "unk_token", "pad_token", "seq_token", "cls_token"]

    def __init__(self, **kwargs):
        self._pad_token = None
        self._unk_token = None
        self._bos_token = None
        self._eos_token = None
        self._seq_token = None
        self._cls_token = None

        for key, value in kwargs.items():
            if key in self.SPECIAL_TOKENS_ATTRIBUTES:
                assert isinstance(value, str) 
                setattr(self, key, value)

    @property
    def pad_token(self):
        if self._pad_token is None:
            logger.error("Using pad_token, but it is not set yet.")
        return self._pad_token

    @property
    def unk_token(self):
        if self._unk_token is None:
            logger.error("Using unk_token, but it is not set yet.")
        return self._unk_token

    @property
    def bos_token(self):
        if self._bos_token is None:
            logger.error("Using bos_token, but it is not set yet.")
        return self._bos_token

    @property
    def eos_token(self):
        if self._eos_token is None:
            logger.error("Using eos_token, but it is not set yet.")
        return self._eos_token

    @property
    def seq_token(self):
        if self._seq_token is None:
            logger.error("Using seq_token, but it is not set yet.")
        return self._seq_token

    @property
    def cls_token(self):
        if self._cls_token is None:
            logger.error("Using cls_token, but it is not set yet.")
        return self._cls_token

    @pad_token.setter
    def pad_token(self, value):
        self._pad_token = value

    @unk_token.setter
    def unk_token(self, value):
        self._unk_token = value

    @bos_token.setter
    def bos_token(self, value):
        self._bos_token = value

    @eos_token.setter
    def eos_token(self, value):
        self._eos_token = value

    @seq_token.setter
    def seq_token(self, value):
        self._seq_token = value

    @cls_token.setter
    def cls_token(self, value):
        self._cls_token = value

    def convert_tokens_to_ids(self, tokens):
        """ Converts a single token, or a sequence of tokens, (str/unicode) in a single integer id
            (resp. a sequence of ids), using the vocabulary.
        """
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        ids = []
        for token in tokens:
            ids.append(self._convert_token_to_id(token))
        return ids

    def _convert_token_to_id(self, token):
        raise NotImplementedError

    def convert_ids_to_tokens(self, ids):
        """ Converts a single index or a sequence of indices (integers) in a token "
            (resp.) a sequence of tokens (str/unicode), using the vocabulary and added tokens.
        """
        if isinstance(ids, int):
            return self._convert_id_to_token(ids)
        tokens = []
        for index in ids:
            tokens.append(self._convert_id_to_token(index))
        return tokens

    def _convert_id_to_token(self, index):
        raise NotImplementedError

class SLUWordTokenizer(Tokenizer):
    """
    special tokens: <bos>, <eos>, <unk>, <pad>, ...
    normal tokens: hello, world, ...
    """
    def __init__(self, bos_eos=False, lowercase=False, bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', seq_token='<seq>', cls_token='<cls>'):
        super(SLUWordTokenizer, self).__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, pad_token=pad_token, seq_token=seq_token, cls_token=cls_token)
        
        self.lowercase = lowercase
        
        self.vocab_size = 0
        self.token_to_id = collections.OrderedDict()
        #self.id_to_token = collections.OrderedDict()
        self.special_tokens = set()
        for token in (self.pad_token, self.unk_token, self.bos_token, self.eos_token, self.seq_token, self.cls_token):
            self.token_to_id[token] = self.vocab_size
            self.vocab_size += 1
            self.special_tokens.add(token)

    def get_vocab_size(self,):
        return self.vocab_size

    def read_word2vec_inText(self, w2v_file, device=None):
        '''be careful when setting lowercase=True'''
        special_token_embeddings = {}
        normal_tokens = collections.OrderedDict()
        normal_token_embeddings = []
        with open(w2v_file, 'r') as f:
            head = f.readline().strip()
            word_num, emb_dim = [int(value) for value in head.split(' ')]
            for line in f:
                line = line.strip('\n\r')
                items = line.split(' ')
                word = items[0]
                if self.lowercase:
                    word = word.lower()
                vector = [float(value) for value in items[1:] if value != ""]
                if word in self.special_tokens:
                    special_token_embeddings[word] = vector
                elif word not in normal_tokens:
                    idx = len(normal_tokens)
                    normal_tokens[word] = idx
                    normal_token_embeddings.append(vector)
        assert len(normal_tokens) == len(normal_token_embeddings)
        normal_token_embeddings = torch.tensor(normal_token_embeddings, dtype=torch.float, device=device)

        new_token_to_id = collections.OrderedDict()
        self.vocab_size = 0
        token_out_of_pretrained_emb_num = 0
        for token in self.token_to_id:
            if token not in normal_tokens:
                new_token_to_id[token] = self.vocab_size
                self.vocab_size += 1
                token_out_of_pretrained_emb_num += 1
        for token in normal_tokens:
            new_token_to_id[token] = self.vocab_size
            self.vocab_size += 1

        self.token_to_id = new_token_to_id
        self.id_to_token = collections.OrderedDict([(ids, tok) for tok, ids in self.token_to_id.items()])

        new_special_token_embeddings = {}
        for token in special_token_embeddings:
            new_special_token_embeddings[self.token_to_id[token]] = torch.tensor(special_token_embeddings[token], dtype=torch.float, device=device)
        
        return new_special_token_embeddings, len(normal_tokens), normal_token_embeddings, token_out_of_pretrained_emb_num

    def save_vocab(self, vocab_file):
        save_vocab_file(self.token_to_id, vocab_file)

    def read_vocab(self, vocab_file):
        vocab = read_vocab_file(vocab_file)
        for token in vocab:
            if token not in self.token_to_id:
                self.token_to_id[token] = self.vocab_size
                self.vocab_size += 1
        self.id_to_token = collections.OrderedDict([(ids, tok) for tok, ids in self.token_to_id.items()])

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.token_to_id.get(token, self.token_to_id.get(self.unk_token))
	
    def _convert_id_to_token(self, index):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.id_to_token.get(index, self.unk_token)
    
    def tokenize(self, text):
        """
        For example:
            input = "unaffable"
            output = ["unaffable"]
        """
        return whitespace_tokenize(text, lowercase=self.lowercase)

class SLUCharTokenizer(Tokenizer):
    """
    special tokens: <bos>, <eos>, <unk>, <pad>, ...
    normal tokens: hello, world, ...
    """
    def __init__(self, bos_eos=False, lowercase=False, bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>', seq_token='<seq>', cls_token='<cls>'):
        super(SLUCharTokenizer, self).__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, pad_token=pad_token, seq_token=seq_token, cls_token=cls_token)
        
        self.lowercase = lowercase
        
        self.vocab_size = 0
        self.token_to_id = collections.OrderedDict()
        #self.id_to_token = collections.OrderedDict()
        self.special_tokens = set()
        for token in (self.pad_token, self.unk_token, self.bos_token, self.eos_token, self.seq_token, self.cls_token):
            self.token_to_id[token] = self.vocab_size
            self.vocab_size += 1
            self.special_tokens.add(token)

    def get_vocab_size(self,):
        return self.vocab_size

    def save_vocab(self, vocab_file):
        save_vocab_file(self.token_to_id, vocab_file)

    def read_vocab(self, vocab_file):
        vocab = read_vocab_file(vocab_file)
        for token in vocab:
            if token not in self.token_to_id:
                self.token_to_id[token] = self.vocab_size
                self.vocab_size += 1
        self.id_to_token = collections.OrderedDict([(ids, tok) for tok, ids in self.token_to_id.items()])

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.token_to_id.get(token, self.token_to_id.get(self.unk_token))
	
    def _convert_id_to_token(self, index):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.id_to_token.get(index, self.unk_token)
    
    def tokenize(self, text):
        """
        For example:
            input = "unaffable"
            output = ['u', 'n', 'a', 'f', 'f', 'a', 'b', 'l', 'e']
        """
        output = []
        for word in whitespace_tokenize(text, lowercase=self.lowercase):
            output.append(tuple(word))
        return output

class SLUOutputVocab(Tokenizer):

    def __init__(self, vocab_data_storage, no_special_labels=False, bos_eos=False, bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>'):
        super(SLUOutputVocab, self).__init__(bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, pad_token=pad_token)
        
        self.label_to_id = collections.OrderedDict()
        self.special_labels = {self.pad_token, self.unk_token, self.bos_token, self.eos_token}
        if no_special_labels:
            self.vocab_size = 0
        else:
            self.label_to_id[self.pad_token] = 0
            self.label_to_id[self.unk_token] = 1
            if bos_eos:
                self.label_to_id[self.bos_token] = 2
                self.label_to_id[self.eos_token] = 3
                self.vocab_size = 4
            else:
                self.vocab_size = 2

        assert type(vocab_data_storage) in {str, list, tuple, dict, set}
        if type(vocab_data_storage) is str:
            vocab = read_vocab_file(vocab_data_storage)
        else:
            vocab = vocab_data_storage
        for label in vocab:
            if label not in self.label_to_id:
                self.label_to_id[label] = self.vocab_size
                self.vocab_size += 1

        self.id_to_label = collections.OrderedDict([(ids, lab) for lab, ids in self.label_to_id.items()])

    def get_vocab_size(self,):
        return self.vocab_size

    def save_vocab(self, vocab_file):
        save_vocab_file(self.label_to_id, vocab_file)

    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.label_to_id.get(token, self.label_to_id.get(self.unk_token))
    
    def _convert_id_to_token(self, index):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.id_to_label.get(index, self.unk_token)

class BIOTagVocab(SLUOutputVocab):
    def __init__(self, vocab_file, no_special_labels=False, bos_eos=False, bos_token='<s>', eos_token='</s>', unk_token='<unk>', pad_token='<pad>'):
        super(BIOTagVocab, self).__init__(vocab_file, no_special_labels=no_special_labels, bos_eos=bos_eos, bos_token=bos_token, eos_token=eos_token, unk_token=unk_token, pad_token=pad_token)
        
        self.label_id_to_bio_and_name_ids = collections.OrderedDict()
        self.bio_to_id = collections.OrderedDict()
        self.special_name_to_id = collections.OrderedDict()
        self.normal_name_to_id = collections.OrderedDict()
        self.bio_of_labels = []
        self.name_of_labels = []
        for lab in self.label_to_id:
            label_id = self.label_to_id[lab]
            if lab in {bos_token, eos_token, unk_token, pad_token}:
                bio, name = 'O', lab
            elif lab == 'O':
                bio, name = 'O', '<O>'
            else:
                bio, name = lab.split('-', 1)
            if bio not in self.bio_to_id:
                self.bio_to_id[bio] = len(self.bio_to_id)
            if name in {bos_token, eos_token, unk_token, pad_token, '<O>'}:
                if name not in self.special_name_to_id:
                    self.special_name_to_id[name] = len(self.special_name_to_id)
            else:
                if name not in self.normal_name_to_id:
                    self.normal_name_to_id[name] = len(self.normal_name_to_id)
            self.bio_of_labels.append(self.bio_to_id[bio])
            self.name_of_labels.append(name)
        for idx, name in enumerate(self.name_of_labels):
            if name in {bos_token, eos_token, unk_token, pad_token, '<O>'}:
                self.name_of_labels[idx] = self.special_name_to_id[name]
            else:
                self.name_of_labels[idx] = len(self.special_name_to_id) + self.normal_name_to_id[name]

    def save_vocab(self, vocab_file, bio_vocab_file, special_name_vocab_file, normal_name_vocab_file):
        save_vocab_file(self.label_to_id, vocab_file)
        save_vocab_file(self.bio_to_id, bio_vocab_file)
        save_vocab_file(self.special_name_to_id, special_name_vocab_file)
        save_vocab_file(self.normal_name_to_id, normal_name_vocab_file)

    def get_bio_tensor_and_selected_slot_indexes(self, device=None):
        bio_one_hot = torch.eye(len(self.bio_to_id))
        bio_tensor = torch.index_select(bio_one_hot, 0, torch.tensor(self.bio_of_labels))
        bio_tensor = bio_tensor.to(device=device)
        selected_slot_indexes = torch.tensor(self.name_of_labels, device=device)
        return bio_tensor, selected_slot_indexes
