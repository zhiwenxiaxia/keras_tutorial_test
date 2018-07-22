'''
This is the exercises from the book titled 'Deep Learning with Python Keras'
Chapter 3: Getting started with neural networks
Section 6.2: Understanding recurrent nerual networks
section 6.2.1: A first recurrent layer in keras
reimplymented by zhiwen
date: 21-Jul-2018
'''

from keras import models, layers, losses, optimizers, metrics
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

# set the results file path
sectionnum = '6_2'
cur_work_path = os.getcwd()
res_path = '{}/res_c{}'.format(cur_work_path,sectionnum)
if not os.path.exists(res_path):
    os.mkdir(res_path)

# params
max_features = 10000 # number of words to condier as features
maxlen = 500 # cut texts after this number of words
batchsize = 32

# loading and prepare the data
print('loading data ...')
(input_train, y_train),(input_test, y_test) = imdb.load_data(num_words=max_features)
print('len of train: ', len(input_train))
print('len of test: ', len(input_test))

print('shape and type of input_train before padding: ', input_train.shape, type(input_train))
input_train = sequence.pad_sequences(input_train,maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('shape and type of input_train after padding: ', input_train.shape, type(input_train))
# crate the model
def CreateModel(rnn = layers.SimpleRNN, hiddensize=32):
    model = models.Sequential()
    model.add(layers.Embedding(max_features,32))
    model.add(rnn(hiddensize))
    model.add(layers.Dense(1,activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    return model

history_record=[]
rnnMethod = [layers.SimpleRNN, layers.LSTM, layers.GRU]

for rnn in rnnMethod:
    model = CreateModel(rnn)
    history = model.fit(input_train,y_train,epochs=10,batch_size=512,validation_split=0.2)
    history_record.append(history)


def plot_on_axis(ax,history,rnnMode,key1='loss',key2='val_loss'):
    value1 = history.history[key1]
    value2 = history.history[key2]
    epochs = range(1,len(value1)+1)
    ax.plot(epochs, value1, 'b', label=key1)
    ax.plot(epochs, value2, 'r', label=key2)
    ax.legend()
    ax.grid(True)
    ax.set_ylabel(str(rnnMode))

plt.figure(figsize=(8,4*len(rnnMethod)))
gs = gridspec.GridSpec(len(rnnMethod),2)
for i, history in enumerate(history_record):
    ax1 = plt.subplot(gs[i,0])
    plot_on_axis(ax1,history,rnnMethod[i],key1='loss',key2='val_loss')
    ax2 = plt.subplot(gs[i,1])
    plot_on_axis(ax2,history,rnnMethod[i],key1='acc',key2='val_acc')
plt.savefig('{}/results_of_different_rnn_nets.png'.format(res_path))
plt.show()



#
