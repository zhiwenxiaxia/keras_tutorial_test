'''
This is the exercises from the book titled 'Deep Learning with Python Keras'
Chapter 3: Getting started with neural networks
Section 6.3: Advanced usage of recurrent nerual networks
section 6.3.1: A temperature forecasting problem
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
sectionnum = '6_4'
cur_work_path = os.getcwd()
res_path = '{}/res_c{}'.format(cur_work_path,sectionnum)
if not os.path.exists(res_path):
    os.mkdir(res_path)

# loading and prepare the data
data_dir = './jena_climate'
fname = os.path.join(data_dir,'jena_climate_2009_2016.csv')
f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]
print(header)
print(len(lines))
# parsing the data
float_data = np.zeros((len(lines), len(header)-1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i,:] = values

# plotting the temperature time series
# temp = float_data[:,1]
# plt.figure(figsize=(6,10))
# plt.subplot(211)
# plt.plot(range(len(temp)),temp)
# plt.xlabel('epochs')
# plt.ylabel('temperature deg')
# plt.title('the whole temperatures')
# plt.subplot(212)
# plt.plot(range(1440),temp[:1440])
# plt.xlabel('epochs')
# plt.ylabel('temperature deg')
# plt.title('one day temperatures')
# plt.savefig('{}/temperature.png'.format(res_path))

# preparing the data
train_size = 200000
mean = float_data[:train_size].mean(axis=0)
float_data -= mean
std = float_data[:train_size].std(axis=0)
float_data /= std

def generator(data, lookback, delay, min_index, max_index,shuffle=False, batch_size=128, step=6):
    if max_index is None:
        max_index = len(data)-delay-1
    i = min_index + lookback

    while 1:
        if shuffle:
            rows = np.random.randint(min_index+lookback, max_index, batch_size)
        else:
            if i+batch_size>=max_index:
                i = min_index+lookback
            rows = np.arange(i,min(i+batch_size, max_index))
            i += len(rows)

        samples = np.zeros((len(rows),lookback//step,data.shape[-1]))

        targets = np.zeros((len(rows),))

        for j, row in enumerate(rows):
            indices = range(row-lookback,row,step)
            samples[j] = data[indices]
            targets[j] = data[row+delay][1]
        yield samples, targets


# preparing the training, validation and  test generator
lookback = 720
step = 3
delay = 144
batch_size = 512

train_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=0,
                      max_index=200000,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)
val_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=200001,
                      max_index=300000,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)
test_gen = generator(float_data,
                      lookback=lookback,
                      delay=delay,
                      min_index=300001,
                      max_index=None,
                      shuffle=True,
                      step=step,
                      batch_size=batch_size)

# compute the val and test steps
train_steps  = (200000-lookback)//batch_size
val_steps = (300000-200001-lookback)//batch_size
test_steps = (len(float_data)-300001-lookback)//batch_size

def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds-targets))
        batch_maes.append(mae)
    return np.mean(batch_maes)

print(evaluate_naive_method()*std[1])

# training and evaluating a densly-connected model using the data generator

def CreateDenselyConnectModel():
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(lookback//step, float_data.shape[-1])))
    model.add(layers.Dense(32,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer=optimizers.RMSprop(), loss='mae')
    return model

# # train and validating using densely_connected_model
# model = CreateDenselyConnectModel()
# plot_model(model=model,
#           to_file='{}/densely_connected_model.png'.format(res_path),
#           show_shapes=True)
#
# history = model.fit_generator(train_gen,steps_per_epoch=train_steps,
#                     epochs=20,validation_data=val_gen,
#                     validation_steps=val_steps)
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1,len(loss)+1)
# plt.figure()
# plt.plot(epochs,val_loss,'r',label='val_loss')
# plt.plot(epochs,loss,'b',label='loss')
# plt.legend()
# plt.grid(True)
# plt.savefig('{}/results_of_densely_connected_model.png'.format(res_path))

# RNN networks
def CreateGRUModel(rnn=layers.GRU,
                   rnn_units=[32,64],
                   dropout=0.,
                   recurent_dropout=0.):
    model = models.Sequential()
    for i, unit in enumerate(rnn_units):

        if i<len(rnn_units)-1:
            Is_return_sequences = True
        else:
            Is_return_sequences = False

        if i is 0:
            model.add(rnn(unit,
                          dropout=dropout,
                          recurrent_dropout=recurent_dropout,
                          return_sequences=Is_return_sequences,
                          input_shape=(None, float_data.shape[-1])))
        else:
            model.add(rnn(unit,
                          dropout=dropout,
                          recurrent_dropout=recurent_dropout,
                          return_sequences=Is_return_sequences))
    model.add(layers.Dense(1))
    model.compile(optimizer=optimizers.RMSprop(), loss='mae')
    return model

def CreateCONVGRUModel():
    model = models.Sequential()
    model.add(layers.Conv1D(32,5,activation='relu',input_shape=(None, float_data.shape[-1])))
    model.add(layers.MaxPool1D(3))
    model.add(layers.Conv1D(32,5,activation='relu'))
    model.add(layers.GRU(32,dropout=0.1,recurrent_dropout=0.5))
    model.add(layers.Dense(1))
    print(model.summary())
    plot_model(model=model,
              to_file='{}/{}.png'.format(res_path,'conv_gru_model'),
              show_shapes=True)
    model.compile(optimizer=optimizers.RMSprop(), loss='mae')
    return model

# run one rnn net and show the prediction results
model = CreateCONVGRUModel()
history = model.fit_generator(train_gen,steps_per_epoch=train_steps,
                              epochs=10,validation_data=val_gen,
                              validation_steps=val_steps,
                              use_multiprocessing=True)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(loss)+1)
plt.figure()
plt.plot(epochs,val_loss,'r',label='val_loss')
plt.plot(epochs,loss,'b',label='loss')
plt.legend()
plt.grid(True)
plt.savefig('{}/results_of_conv_rnn.png'.format(res_path))
plt.show()


predictions = model.predict_generator(test_gen,steps=test_steps)

test_targets = []
for step in range(test_steps):
    sample, targets = next(test_gen)
    test_targets.append(targets)
len(test_targets)
len(test_targets[0])

plt.figure()
plt.plot(range(1,101),test_targets[0][:100],'r-',label='target')
plt.plot(range(1,101),predictions[:100],'b-',label='predict')
plt.legend()
plt.savefig('{}/prediction_vs_targets.png'.format(res_path))
plt.show()

# print(evaluate_naive_method()*std[1])
#
# test_name=['GRU_32_dropout','GRU_64_dropout','LSTM_32_dropout','LSTM_64_dropout','GRU_32_32_dropout']
# test_RNN = [CreateGRUModel(rnn=layers.GRU,rnn_units=[32],dropout=0.2,recurent_dropout=0.5),
#             CreateGRUModel(rnn=layers.GRU,rnn_units=[64],dropout=0.2,recurent_dropout=0.5),
#             CreateGRUModel(rnn=layers.LSTM,rnn_units=[32],dropout=0.2,recurent_dropout=0.5),
#             CreateGRUModel(rnn=layers.LSTM,rnn_units=[64],dropout=0.2,recurent_dropout=0.5),
#             CreateGRUModel(rnn=layers.GRU,rnn_units=[32,32],dropout=0.2,recurent_dropout=0.5)]
#
# # Run the rnn networks
# history_record = []
# for modelname,model in zip(test_name,test_RNN):
#     plot_model(model=model,
#               to_file='{}/{}.png'.format(res_path,modelname),
#               show_shapes=True)
#     history = model.fit_generator(train_gen,steps_per_epoch=train_steps,
#                                   epochs=20,validation_data=val_gen,
#                                   validation_steps=val_steps)
#     history_record.append(history)
#
#
#
# def plot_on_axis(ax,history,rnnMode,key1='loss',key2='val_loss'):
#     value1 = history.history[key1]
#     value2 = history.history[key2]
#     epochs = range(1,len(value1)+1)
#     ax.plot(epochs, value1, 'b', label=key1)
#     ax.plot(epochs, value2, 'r', label=key2)
#     ax.legend()
#     ax.grid(True)
#     ax.set_ylabel(str(rnnMode))
#
#
# # show the results
# plt.figure(figsize=(6,6*len(test_name)))
# gs = gridspec.GridSpec(len(test_name),1)
# for i, history in enumerate(history_record):
#     ax1 = plt.subplot(gs[i,0])
#     plot_on_axis(ax1,history,test_name[i],key1='loss',key2='val_loss')
#     # ax2 = plt.subplot(gs[i,1])
#     # plot_on_axis(ax2,history,test_name[i],key1='acc',key2='val_acc')
# plt.savefig('{}/results_comparisin_of_different_rnn_net_work.png'.format(res_path))
# plt.show()
