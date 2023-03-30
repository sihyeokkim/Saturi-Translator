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

class trans_former_config :
    def __init__(self,vocab_small=True) :
        self.n_layers =6
        self.d_model=512
        self.n_heads=8
        self.d_ff=2048
        self.src_vocab_size=16009 if vocab_small else 8009
        self.tgt_vocab_size=16009 if vocab_small else 8009
        self.pos_len= 512
        self.dropout=0.2
        self.shared=True
        
class tokenizer_config(trans_former_config) :
    def __init__(self, token_type = 'cmsp') :
        super().__init__()
        self.token_type = token_type
        self.token_vocab_size = 16000 if self.src_vocab_size == 16009 else 8009
        self.enc_token_load_model = data_path + f'spm_enc_spm{self.token_vocab_size}.model'
        self.dec_token_load_model = data_path + f'spm_dec_{self.token_type}{self.token_vocab_size}.model'

class translator(tokenizer_config) :
    
    def __init__(self) :
        super().__init__()
        self.enc_tokenizer = spm.SentencePieceProcessor()
        self.enc_tokenizer.Load(self.enc_token_load_model)
        self.dec_tokenizer = spm.SentencePieceProcessor()
        self.dec_tokenizer.Load(self.dec_token_load_model)
        self.model = Transformer(self.n_layers,
                                 self.d_model,
                                 self.n_heads,
                                 self.d_ff,
                                 self.src_vocab_size,
                                 self.tgt_vocab_size,
                                 self.pos_len,
                                 self.dropout,
                                 self.shared)
        self.model.load_weights()
        
    def translate(self, input_txt) :
        ids, result, enc_attns, dec_attns, dec_enc_attns = evaluate(input_txt, self.model, self.enc_tokenizer, self.dec_tokenizer)
        return result