
"""
BLUE値:似ている度合を測定する尺度：BLUE

BP_BLEU: 翻訳文が正解分と比較して短"い場合"に用いるペナルティ項
これは翻訳文が正解文より短いときに適合率が上がってしまうから.
適合率：分母を翻訳でみるもの．
r :正解文の単語数
c :翻訳文の単語数

C > r = 1
e^(1 - r/c)
1に近い方が精度が高い
#http://phontron.com/paper/neubig13nl212.pdf
Nは基本4でやることが多い

"""

import  numpy as np
import itertools
import sys

def ngram(words,n):
    return list(zip(*(words[i:] for i in range(n))))


#Excel data

def BLUE(filename):
    test_data = [["This", "is","the","data"],["a","a"]]
    trans_data = [["This", "is","the","data"],["a"]]
    BP_BLUE = 0
    counts = {}
    test_data_length = len(list(itertools.chain.from_iterable(test_data)))
    trans_data_length = len(list(itertools.chain.from_iterable(trans_data)))

    if (test_data_length  > trans_data_length):
        BP_BLUE = np.exp(1-test_data_length/trans_data_length)
    else :
        BP_BLUE = 1


    one_trans_gram = ngram(trans_data, 1)
    n_maxlength = max([len(i) for i in trans_data])
    print(n_maxlength)
    ncounts = [0]*n_maxlength
    a_length = [0]*n_maxlength
    p = 0

    for k in range(n_maxlength):
        for [trans_sentence,test_sentence] in zip(trans_data ,test_data):
            n_trans_gram = ngram(trans_sentence, k+1)
            n_tests_gram = ngram(test_sentence, k+1)
            a_length[k] += len(n_trans_gram)
            for i,tran_gram in enumerate(n_trans_gram):
                for j,test_gram in enumerate(n_tests_gram):
                    if(tran_gram == test_gram):
                        n_tests_gram.pop(j)
                        ncounts[k] += 1
                        break

        p +=  np.log(ncounts[k]/a_length[k])


    BLUE = BP_BLUE * np.exp(p)

    print(BLUE)
    with open("./ResultBLEU.txt",mode='w') as f:
        f.write("BLUE"+str(BLUE))



if __name__ == '__main__':

    if '-f' in sys.argv and len(sys.argv) >= 3:
        #print(sys.argv[2])
        BLUE(sys.argv[2])
    else:
        print("引数に-fをつけてください.")