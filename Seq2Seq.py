# coding=utf-8

"""
LSTMの別で学習できるように作成するために作成するスクリプト作成する

"""

import matplotlib.pylab as plt
import  numpy as np
from  keras.models import load_model
import datetime
from keras.optimizers  import SGD, RMSprop, Adam
from keras.layers import  LSTM,Dense,Input, RepeatVector
from keras.models import Model
from keras.callbacks import TensorBoard
import pandas as pd
from keras.preprocessing.text import one_hot
from keras.utils.np_utils import to_categorical
from keras.layers.core   import Flatten,Activation
from keras.utils.vis_utils import plot_model
import pickle
import os
import glob
from keras_contrib.utils import save_load_utils

#TODO:どれくらいまわせばいいのか，語彙いくつでどれだけ回せばいいのかを調べる
#評価方法を調べる

#kerasｎついてModel(a,b)でaを入力としてbを出力するためにのそうを含まれる．
#https://keras.io/ja/models/about-keras-models/




def Readdata(filename):
    df = pd.read_csv(filename)
    datas = df["data"]
    label_datas = df["labeldata"]
    datas = datas.values
    label_datas = label_datas.values

    def Append(datas):
        max_vector_lenth = 5 + 2#length 決める
        textdatas = np.empty((0,max_vector_lenth), int)#ここで長さは同じにしてる

        for data in datas:
            text = np.array([])
            data_list = str(data).split()
            data_length = len(data_list)
            #for i in range(max_vector_lenth):
            for i, word in enumerate(data_list):
                 #try:
                #    word = data_list[i]
                #except IndexError:
                #    word = 0
                if(i == 0):
                    text = np.append(text,"<BOS>")
                    text = np.append(text, word)
                elif(i >= data_length - 1):
                    text = np.append(text, word)
                    text = np.append(text, "<EOS>")
                else:
                    text = np.append(text, word)
            #print(len(text))
            if (max_vector_lenth - len(text) ) > 0:#ここでone_hot作成する．+1 はEOSです
                append_null= max_vector_lenth - (len(text))
                a =np.zeros(append_null,dtype=int)
                text = np.insert(text,-1,a)
            textdatas = np.append(textdatas,[text],axis = 0)
        return textdatas

    datas =Append(datas)
    label_datas = Append(label_datas)
    #print(datas)
    return datas,label_datas



def Onehot(datas):
    sentence_length =len(datas[0])
    listdata = ','.join(datas.flatten())
    onehot_data = one_hot(listdata,13,lower=True,split=',')
    onehot_data_flat = to_categorical(onehot_data)#数字をベクトルに変更する
    onehot_datas = []
    ndata = 0
    for data in datas:
        temp =[]
        for i in range(len(data)):
            temp.append(onehot_data_flat[ndata + i])
        onehot_datas.append(temp)
        ndata += len(data)
    return np.array(onehot_datas)


#target も作成できるようにする
def OnehotMakedata(filedatalist,picklefilename):
    sentence_length = len(filedatalist[0])
    print(sentence_length)
    c_f = {}  # {}で辞書オブジェクトを作成する c_fはindexの場所を指す
    depth = 100  # Vocablaryによるが#depth位以下の単語を削る
    "Vocaburary辞書を作成する"
    if (os.path.isfile("./" + picklefilename) == False):
        for data in filedatalist:
            for word in data:
                if c_f.get(word) is None:  # 要素がないなら作成する. c番目の値を0にする
                    # https: // note.nkmk.me / python - dict - list - values /
                    c_f[word] = 0
                c_f[word] += 1  # 文字列を多くする.
        # sorted タプルをソートできる. 数が多い順に並び替えて表示
        #for e, (c, f) in enumerate(sorted(c_f.items(), key=lambda x: x[1] * -1)):
        #    print(e, c, f)
        c_i = {c: e for e, (c, f) in enumerate(sorted(c_f.items(), key=lambda x: x[1] * -1)[:depth])}
        c_i["0"] = len(c_i)
        open(picklefilename, "wb").write(pickle.dumps(c_i))
    else :
        c_i = pickle.load(open(picklefilename, "rb"))

    #辞書ロードし，それをOnehotに変更する.Readdata
    onehot_vector = []
    Next_onehot_vector = []
    for i,data in enumerate(filedatalist):
        xd = [[0.] * len(c_i) for _ in range(sentence_length)]
        Next_target =  [[0.] * len(c_i) for _ in range(sentence_length)]
        for l,word in enumerate(data):
            xd[l][c_i[word]] = 1.
            if l > 0 :
                Next_target[l - 1][c_i[word]] = 1.
            elif l >= len(data) :
                Next_target[l][c_i["<EOS>"]] = 1.

        onehot_vector.append(np.array(list(xd)))
        Next_onehot_vector.append(np.array(list(Next_target)))

    print("shapeのデータ",np.array(onehot_vector).shape)

    return np.array(onehot_vector),np.array(Next_onehot_vector)

traindata_filename = "reverseData5length.csv"
train_V_picklename ="Vocabulary.pkl"
testdata_file_name = "reverseData5length.csv"

train_data,label_data = Readdata(traindata_filename)
train_oh_datas,Next_oh_target = OnehotMakedata(train_data,train_V_picklename)
train_ohl_datas,Next_ohl_target = OnehotMakedata(label_data,train_V_picklename)


#train_onehot_datas = Onehot(train_data)
#train_label_datas = Onehot(label_data)
#test_datas, test_label_datas = Readdata(testdata_file_name)
#test_onehot_datas = OnehotMakedata(test_datas,train_V_picklename)
#test_onehotl_datas = OnehotMakedata(test_label_datas,train_V_picklename)


def Predict(autoencoder,V_picklename):
    c_i = pickle.loads(open(V_picklename, "rb").read())
    i_c = {i: c for c, i in c_i.items()} #これ好き

    model = sorted(glob.glob("./model/*.h5")).pop(0)
    print("loaded model is ", model)
    model = load_model(model)

    encoder_inputs = model.input[0]  # input_1
    encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_inputs = model.input[1]  # input_2
    decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
    decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[3]
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[4]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    def decode_sequence(input_seq):
        b = np.reshape(input_seq, (1,7,11))
        states_value = encoder_model.predict([b])
        target_seq = np.zeros((1, 1, 11))
        target_seq[0, 0, c_i['<EOS>']] = 1.

        stop_condition = False
        decoded_sentence = ''
        seq_length = 7
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)

            sample_char = i_c[np.argmax(output_tokens)]
            decoded_sentence += sample_char
            if (sample_char == '<EOS>' or
                    len(decoded_sentence) > seq_length):
                stop_condition = True
            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, 11))
            target_seq[0, 0, np.argmax(output_tokens)] = 1.
            # Update states
            states_value = [h, c]
        return decoded_sentence


    for i,(test_data,one_hot) in enumerate(zip(train_data,train_oh_datas)):
        print("------------------------------------------------------")
        print("データ番号:",i+1,"テストデータ:",test_data)
        decoder_sentence=decode_sequence(one_hot)
        print("decoder",decoder_sentence)








if __name__ == '__main__':

    """
    a = np.random.random(100)
    x = np.array([np.sin([p for p in np.arange(0, 0.8, 0.1)] + aa) for aa in a])
    x = np.array(x).reshape(100,8,1)#data数,length,dim
    y = -x
    print(x.shape) #(1000,8) 8がlength 1000がデータ・セット
    """

    #print(sentence_length)
    #print( vocabulary)
    ndata = 100

    max_input_length = 7
    latent_dim = 11
    sentence_length = train_oh_datas.shape[1]
    vocabulary = train_oh_datas.shape[2]

    encoder_inputs = Input(shape=(None,vocabulary))
    encoder = LSTM(latent_dim,return_state=True)
    encoder_outputs,state_h,state_c = encoder(encoder_inputs)
    encoder_states =[state_h,state_c]

    #set Decoder
    decoder_inputs = Input(shape=(None,vocabulary))
    decoder_lstm = LSTM(latent_dim,return_state= True,return_sequences=True)
    decoder_outputs,_,_=decoder_lstm(decoder_inputs,initial_state=encoder_states)
    decoder_dense = Dense(vocabulary,activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    #Define the  model that will turn
    model = Model([encoder_inputs,decoder_inputs],decoder_outputs)

    model.compile(optimizer=Adam(), loss='categorical_crossentropy')
    model.summary()
    plot_model(model,show_shapes =True,show_layer_names = True,to_file='model/model.png')#dirも追加可能

    sys="predict"
    if(sys =="fit"):
        tb_cb = TensorBoard(log_dir="log",histogram_freq=1, write_graph=True, write_images=True)
        model.fit([train_oh_datas,train_ohl_datas],Next_ohl_target,epochs=1000,batch_size = 1,
                  callbacks=[tb_cb],validation_data=([train_oh_datas,train_ohl_datas],Next_ohl_target))#acc は正解率
        now = datetime.datetime.today()
        model.save('./model/s2s'+now.month+'_'+now.hour+'_'+now.minute+'.h5')

    elif(sys == "predict"):
        Predict(model,train_V_picklename)


    # 未学習のデータでテスト
    #x_test = np.array([np.sin([[p] for p in np.arange(0, 0.8, 0.1)] + aa) for aa in np.arange(0, 1.0, 0.1)])
    #print(model.evaluate(x_test, y_test, batch_size=32))
    # 未学習のデータで生成
    #predicted = model.predict(x_test, batch_size=32)


