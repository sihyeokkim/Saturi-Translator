import os
import sys


sys.path.insert(0,os.getenv('HOME') + '/aiffel/saturi/MODEL/') # vanilla transformer 파일경로에서 불러오기 위해 설정
sys.path.insert(0,os.getenv('HOME') + '/aiffel/saturi/PRE/')
sys.path.insert(0,os.getenv('HOME') + '/aiffel/saturi/POST/')

from translate import translator

if __name__ == '__main__' :
    
    token_type = input('select - cmsp, msp, spm : \n')
    vocab_size = input('small_vocab? Yes or No : \n')
    if (vocab_size == 'yes') | (vocab_size == 'Yes') | (vocab_size == 'y') | (vocab_size == 'Y') :
        vocab_size = True
    translator = translator(token_type, vocab_size)

    while True :
        print('Add region tag before your input text :')
        print('--------------------------------------------------------------------')
        print('jeju : <jj> , kyengsang : <gs> , chunchung : <cc> , kangwon : <kw> , jeonra : <jd>')
        print('--------------------------------------------------------------------')
        input_txt = input()
        print('**********************************************************************')
        print('input_text : ', input_txt)
        print('translated_text : ',translator.translate(input_txt))
        print('**********************************************************************\n\n')