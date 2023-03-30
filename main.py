import sys
sys.path.insert(0,os.getenv('HOME') + '/aiffel/saturi/MODEL/') # vanilla transformer 파일경로에서 불러오기 위해 설정
sys.path.insert(0,os.getenv('HOME') + '/aiffel/saturi/PRE/')
sys.path.insert(0,os.getenv('HOME') + '/aiffel/saturi/POST/')

from translate import translator

if __name__ == '__main__' :
    
    translator = translator()

    while True :
        print('Add region tag before your input text :')
        print('jeju : <jj> , kyengsang : <gs> , chunchung : <cc> , kangwon : <kw> , jeonra : <jd> ')
        input_txt = input()
        print('input_text : ', input_txt)
        print('translated_text : ',translator.translate(input_txt))