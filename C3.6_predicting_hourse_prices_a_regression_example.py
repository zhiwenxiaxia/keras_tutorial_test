'''
This is the exercises from the book titled 'Deep Learning with Python Keras'
Chapter 3: Getting started with neural networks
Section 3.6: Predicting house price: A regression example
reimplymented by zhiwen
date: 21-Jul-2018
'''

from keras import models, layers, losses, optimizers, metrics
from keras.datasets import boston_housing
from keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

# set the results file path
sectionnum = '3_6'
cur_work_path = os.getcwd()
res_path = '{}/res_c{}'.format(cur_work_path,sectionnum)
if not os.path.exists(res_path):
    os.mkdir(res_path)

# phase 1 : loading the datasets
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
print('train data shape: {};\n test data shape: {}'.format(train_data.shape,test_data.shape))
print('train_targets shape: {};\n test_targets shape: {}'.format(train_targets.shape,test_targets.shape))
# look at the data labels
train_mean = train_data.mean(axis=0)
train_std = train_data.std(axis=0)
x = range(1,14)
plt.figure(figsize=(8,4))
plt.errorbar(x,train_mean,yerr=train_std,fmt='o',ecolor='r',color='b',elinewidth=2,capsize=4)
plt.xlabel('feature number')
plt.ylabel('mean and std values')
plt.title('features distribution which shows the importance of data normalization')
plt.savefig('{}/raw_data_features_distribution.png'.format(res_path))

# phase 2: preprocess the data
train_data = (train_data-train_mean)/train_std
test_data = (test_data-train_mean)/train_std

# phase 3: create the model
def CreateModel(Hidden_size=[64,64], active_fun='relu'):
    model = models.Sequential()
    for i, hs in enumerate(Hidden_size):
        if i is 0:
            model.add(layers.Dense(hs,activation=active_fun,input_shape=(13,)))
        else:
            model.add(layers.Dense(hs,activation=active_fun))

    model.add(layers.Dense(1))

    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    print(model.summary())

    return model


# phase 4: K-fold validation
k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []
for i in range(k):
    print('processing fold #', i)
    # prepare the train and validation data
    val_data = train_data[i*num_val_samples:(i+1)*num_val_samples]
    val_targets = train_targets[i*num_val_samples:(i+1)*num_val_samples]

    partial_train_data = np.concatenate([train_data[:i*num_val_samples],
                                        train_data[(i+1)*num_val_samples:]],
                                        axis=0)
    partial_train_targets = np.concatenate([train_targets[:i*num_val_samples],
                                        train_targets[(i+1)*num_val_samples:]],
                                        axis=0)
    # create the model
    model = CreateModel(Hidden_size=[64,32],active_fun='relu')
    # train the model and validation
    history = model.fit(partial_train_data,
            partial_train_targets,
            validation_data=(val_data,val_targets),
            epochs=num_epochs,
            batch_size=64,
            verbose=1)
    # append the results
    all_scores.append(history.history['val_mean_absolute_error'])


plt.figure(figsize=(8,6))
plt.plot(all_scores[0],'r',label='fold#0')
plt.plot(all_scores[1],'g',label='fold#1')
plt.plot(all_scores[2],'b',label='fold#2')
plt.plot(all_scores[3],'m',label='fold#3')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('val_mean_absolute_error')
plt.title('F-fold validation results')
plt.savefig('{}/val_mean_absolute_error.png'.format(res_path))
plt.show()



# # show the results
# def plot_on_axis(ax,history,Hidden_size=[64,64],key1='loss',key2='val_loss'):
#     value1 = history.history[key1]
#     value2 = history.history[key2]
#     epochs = range(1,len(value1)+1)
#     ax.plot(epochs, value1, 'b', label=key1)
#     ax.plot(epochs, value2, 'r', label=key2)
#     ax.legend()
#     ax.grid(True)
#     ax.set_ylabel(''.join(str(Hidden_size)))
#
#
# # train and evaluate the model
# valid_size = 1000
# epochs = 20
# x_val = x_train[:valid_size]
# y_val = y_train[:valid_size]
# px_train = x_train[valid_size:]
# py_train = y_train[valid_size:]
#
#
# hidden_sizes = [[128,64],[128,128,64],[128,16,64],[32,16],[256,128],[64,64,64,64]]
#
# history_record=[]
# for hs in hidden_sizes:
#     model = CreateModel(Hidden_size=hs,classes=46,active_fun='relu')
#     history = model.fit(px_train,
#              py_train,
#              epochs=epochs,
#              batch_size=512,
#              validation_data=(x_val, y_val))
#     history_record.append(history)
#
# # show the results
# print(type(history_record))
#
# plt.figure(figsize=(16,len(history_record)*6))
# gs = gridspec.GridSpec(len(history_record),2)
# for i, history in enumerate(history_record):
#     ax1 = plt.subplot(gs[i,0])
#     plot_on_axis(ax1,history,Hidden_size=hidden_sizes[i],key1='loss',key2='val_loss')
#     ax2 = plt.subplot(gs[i,1])
#     plot_on_axis(ax2,history,Hidden_size=hidden_sizes[i],key1='acc',key2='val_acc')
# plt.savefig('{}/train_and_test_results.png'.format(res_path))
# plt.show()
