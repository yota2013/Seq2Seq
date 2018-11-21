# coding=utf-8

"""
LSTMの別で学習できるように作成するために作成するスクリプト作成する
出力形式

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
import sys

#評価方法を調べる

#kerasｎついてModel(a,b)でaを入力としてbを出力するためにのそうを含まれる．
#https://keras.io/ja/models/about-keras-models/


def Readdata(filename):
    df = pd.read_csv(filename)
    datas = df["data"]
    label_datas = df["labeldata"]

    def Append(datas):

        textdatas = []
        for data in datas:
            text = np.array([])
            data_list = str(data).split(" ")
            data_length = len(data_list)
            for i, word in enumerate(data_list):
                if (i == 0):
                    text = np.append(text, "<BOS>")
                    text = np.append(text, word)
                elif (i >= data_length - 1):
                    text = np.append(text, word)
                    text = np.append(text, "<EOS>")
                else:
                    text = np.append(text, word)
            textdatas.append(text)
        return np.array(textdatas)

    datas = Append(datas)
    label_datas = Append(label_datas)

    return datas, label_datas


#target も作成できるようにする
def OnehotMakedata(filedatalist,picklefilename):
    max_sentence_length = 0
    for word in filedatalist:
        if max_sentence_length < len(word):
            max_sentence_length = len(word)
    sentence_length = max_sentence_length

    print(sentence_length)
    c_f = {}  # {}で辞書オブジェクトを作成する c_fはindexの場所を指す
    depth = 199  # Vocablaryによるが#depth位以下の単語を削る

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

    #print("shapeのデータ",np.array(onehot_vector).shape)

    return np.array(onehot_vector),np.array(Next_onehot_vector)


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
        b = np.reshape(input_seq, (1,20,200))
        states_value = encoder_model.predict([b])
        target_seq = np.zeros((1, 1, 200))
        target_seq[0, 0, c_i['<EOS>']] = 1.

        stop_condition = False
        decoded_sentence = []
        seq_length = 20
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)

            sample_char = i_c[np.argmax(output_tokens)]
            decoded_sentence.append(sample_char)

            if (sample_char == '<EOS>' or
                    len(decoded_sentence) > seq_length):
                stop_condition = True
            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, 11))
            target_seq[0, 0, np.argmax(output_tokens)] = 1.
            # Update states
            states_value = [h, c]
        return decoded_sentence

    inputdata = []
    outputdata = []
    for i, (test_data, one_hot) in enumerate(zip(train_data, train_oh_datas)):
        outputsentence = decode_sequence(one_hot)
        if (len(inputdata) == len(outputdata)):
            inputdata.append(test_data)
            outputdata.append(outputsentence)
        else:
            print("Error")
    return [inputdata,outputdata]












if __name__ == '__main__':

    traindata_filename = "traindata.csv"
    train_V_picklename = "Vocabulary.pkl"
    V_picklename = "Vocabularytrain.pkl"
    testdata_file_name = "traindata.csv"
    print("data Download")
    train_data, label_data = Readdata(traindata_filename)
    print("Onehot Vector")
    train_oh_datas, Next_oh_target = OnehotMakedata(train_data, train_V_picklename)
    train_ohl_datas, Next_ohl_target = OnehotMakedata(label_data, V_picklename)

    ndata = 100
    latent_dim = 11

    vocabulary_encoder = train_oh_datas.shape[2]
    vocabulary_decoder = train_ohl_datas.shape[2]

    now = datetime.datetime.today()

    print("directiry check")
    if os.path.exists("./result") == False:
        os.mkdir("result")
    if os.path.exists("./model") == False:
        os.mkdir("model")
    if os.path.exists("./log") == False:
        os.mkdir("log")



    encoder_inputs = Input(shape=(None,vocabulary_encoder))
    encoder = LSTM(latent_dim,return_state=True)
    encoder_outputs,state_h,state_c = encoder(encoder_inputs)
    encoder_states =[state_h,state_c]

    #set Decoder
    decoder_inputs = Input(shape=(None,vocabulary_decoder))

    decoder_lstm = LSTM(latent_dim,return_state= True,return_sequences=True)
    decoder_outputs,_,_=decoder_lstm(decoder_inputs,initial_state=encoder_states)
    decoder_dense = Dense(vocabulary_decoder,activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    #Define the  model that will turn
    model = Model([encoder_inputs,decoder_inputs],decoder_outputs)

    model.compile(optimizer=Adam(), loss='categorical_crossentropy')
    model.summary()
    plot_model(model,show_shapes =True,show_layer_names = True,to_file='model/model.png')#dirも追加可能

    if("-train" in sys.argv):
        tb_cb = TensorBoard(log_dir="log",histogram_freq=1, write_graph=True, write_images=True)
        model.fit([train_oh_datas,train_ohl_datas],Next_ohl_target,epochs=1000,batch_size = 1,
                  callbacks=[tb_cb],validation_data=([train_oh_datas,train_ohl_datas],Next_ohl_target))#acc は正解率
        model.save('./model/s2s'+now.month+'_'+now.hour+'_'+now.minute+'.h5')
    elif("-predict" in sys.argv):
        inputdatas,outputdatas= Predict(model,train_V_picklename)
        with open("./result/Result" + str(now.month) + '_' + str(now.hour) + '_' + str(now.minute) + ".csv",
                  mode="w") as f:
            f.writelines("Input,output")
            for i, (test_data, outputsentence) in enumerate(zip(inputdatas, outputdatas)):
                for i, word in enumerate(test_data):
                    if (i == 1):
                        f.write(word)
                    elif word != "<BOS>" and word != "<EOS>":
                        f.write(" " + word)
                f.write(",")
                init = 1
                for i, word in enumerate(outputsentence):
                    if word != "<BOS>" and word != "<EOS>":
                        if (init == 1):
                            f.write(word)
                            init = 0
                        else:
                            f.write(" " + word)

                f.write("\n")
    else:
        print("-predか-trainかを引数にしてください.")


    # 未学習のデータでテスト
    #x_test = np.array([np.sin([[p] for p in np.arange(0, 0.8, 0.1)] + aa) for aa in np.arange(0, 1.0, 0.1)])
    #print(model.evaluate(x_test, y_test, batch_size=32))
    # 未学習のデータで生成
    #predicted = model.predict(x_test, batch_size=32)
