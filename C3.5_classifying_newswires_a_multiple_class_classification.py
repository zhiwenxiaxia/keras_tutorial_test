'''
This is the exercises from the book titled 'Deep Learning with Python Keras'
Chapter 3: Getting started with neural networks
Section 3.5: Classifying newswires: A multi-class classification example
reimplymented by zhiwen
date: 20-Jul-2018
'''

from keras import models, layers, losses, optimizers, metrics
from keras.datasets import reuters
from keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

# set the results file path
sectionnum = '3_5'
cur_work_path = os.getcwd()
res_path = '{}/res_c{}'.format(cur_work_path,sectionnum)
if not os.path.exists(res_path):
    os.mkdir(res_path)

# phase 1 : loading the datasets
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
print('train data len: {};\n test data len: {}'.format(len(train_data),len(test_data)))
# look at the data labels
plt.figure(figsize=(10,4))
plt.subplot(121)
n, bins, patches = plt.hist(train_labels, 50, density=True, facecolor='g', alpha=0.75)
plt.ylabel('train_labels_hist')
plt.xlabel('labels')
plt.subplot(122)
plt.hist(test_labels, 50, density=True, facecolor='m', alpha=0.5)
plt.ylabel('test_labels_hist')
plt.xlabel('labels')
plt.grid(True)
plt.savefig('{}/train_and_test_labels_hist.png'.format(res_path))

# phase 2: preprocess the data
def to_one_hot(labels, dim=46):
    res = np.zeros((len(labels),dim))
    for i,l in enumerate(labels):
        res[i,l] = 1.
    return res

def vectorize_sequences(seq, dimension=10000):
    results = np.zeros((len(seq), dimension))
    for i, s in enumerate(seq):
        results[i, s] = 1.
    return results

x_train = vectorize_sequences(train_data)
y_train = to_one_hot(train_labels)

x_test = vectorize_sequences(test_data)
y_test = to_one_hot(test_labels)

# phase 3: create the model
def CreateModel(Hidden_size=[128,64], classes=46, active_fun='relu'):
    model = models.Sequential()
    for i, hs in enumerate(Hidden_size):
        if i is 0:
            model.add(layers.Dense(hs,activation=active_fun,input_shape=(10000,)))
        else:
            model.add(layers.Dense(hs,activation=active_fun))

    model.add(layers.Dense(classes,activation='softmax'))

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
    print(model.summary())

    return model

# show the results
def plot_on_axis(ax,history,Hidden_size=[64,64],key1='loss',key2='val_loss'):
    value1 = history.history[key1]
    value2 = history.history[key2]
    epochs = range(1,len(value1)+1)
    ax.plot(epochs, value1, 'b', label=key1)
    ax.plot(epochs, value2, 'r', label=key2)
    ax.legend()
    ax.grid(True)
    ax.set_ylabel(''.join(str(Hidden_size)))


# train and evaluate the model
valid_size = 1000
epochs = 20
x_val = x_train[:valid_size]
y_val = y_train[:valid_size]
px_train = x_train[valid_size:]
py_train = y_train[valid_size:]


hidden_sizes = [[128,64],[128,128,64],[128,16,64],[32,16],[256,128],[64,64,64,64]]

history_record=[]
for hs in hidden_sizes:
    model = CreateModel(Hidden_size=hs,classes=46,active_fun='relu')
    history = model.fit(px_train,
             py_train,
             epochs=epochs,
             batch_size=512,
             validation_data=(x_val, y_val))
    history_record.append(history)

# show the results
print(type(history_record))

plt.figure(figsize=(16,len(history_record)*6))
gs = gridspec.GridSpec(len(history_record),2)
for i, history in enumerate(history_record):
    ax1 = plt.subplot(gs[i,0])
    plot_on_axis(ax1,history,Hidden_size=hidden_sizes[i],key1='loss',key2='val_loss')
    ax2 = plt.subplot(gs[i,1])
    plot_on_axis(ax2,history,Hidden_size=hidden_sizes[i],key1='acc',key2='val_acc')
plt.savefig('{}/train_and_test_results.png'.format(res_path))
plt.show()
