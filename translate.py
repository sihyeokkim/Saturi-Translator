import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
import numpy as np

import re
import os

import sys
sys.path.insert(0,os.getenv('HOME') + '/aiffel/saturi/MODEL/') # vanilla transformer 파일경로에서 불러오기 위해 설정
sys.path.insert(0,os.getenv('HOME') + '/aiffel/saturi/PRE/')
sys.path.insert(0,os.getenv('HOME') + '/aiffel/saturi/POST/')

from evaluation import evaluate

import sentencepiece as spm
from vanilla_transformer import Transformer, generate_masks
data_path = os.getcwd() + '/notebook/Preprocessing/'
weight_dir = os.getenv("HOME") + '/aiffel/aiffelthon/final_organizing_folder/transformer_cmsp_16009'
class trans_former_config :
    def __init__(self) :
        self.n_layers =6
        self.d_model=512
        self.n_heads=8
        self.d_ff=2048
        self.src_vocab_size= 16009
        self.tgt_vocab_size= 16009
        self.pos_len= 512
        self.dropout=0.2
        self.shared=True
        
class tokenizer_config :
    def __init__(self, token_type = 'cmsp', vocab_small = False) :
        self.token_type = token_type
        self.token_vocab_size = 8009 if vocab_small else 16000
        self.enc_token_load_model = data_path + f'spm_enc_spm{self.token_vocab_size}.model'
        self.dec_token_load_model = data_path + f'spm_dec_{self.token_type}{self.token_vocab_size}.model'
        self.enc_tokenizer = spm.SentencePieceProcessor()
        self.enc_tokenizer.Load(self.enc_token_load_model)
        self.dec_tokenizer = spm.SentencePieceProcessor()
        self.dec_tokenizer.Load(self.dec_token_load_model)
        
class translator(trans_former_config) :
    
    def __init__(self, token_type = 'cmsp', vocab_small=None) :
        super().__init__()
        self.token_type = token_type
        if vocab_small :
            self.src_vocab_size= 8009
            self.tgt_vocab_size= 8009
            self.tokenizer = tokenizer_config(self.token_type, vocab_small)
        else :
            self.tokenizer = tokenizer_config(self.token_type)
        self.model = Transformer(self.n_layers,
                                 self.d_model,
                                 self.n_heads,
                                 self.d_ff,
                                 self.src_vocab_size,
                                 self.tgt_vocab_size,
                                 self.pos_len,
                                 self.dropout,
                                 self.shared)
        self.weight_dir = os.getenv("HOME") + f'/aiffel/aiffelthon/final_organizing_folder/transformer_{self.token_type}_{self.src_vocab_size}'
        self.model.load_weights(self.weight_dir)
        
    def translate(self, input_txt) :
        ids, result, enc_attns, dec_attns, dec_enc_attns = evaluate(input_txt, self.model, self.tokenizer.enc_tokenizer, self.tokenizer.dec_tokenizer)
        return result