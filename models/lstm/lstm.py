
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import sys
import glob
from os import listdir
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


# In[2]:

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
from keras.callbacks import Callback
from keras.models import model_from_json
from keras.optimizers import Adam
from keras.regularizers import WeightRegularizer


# In[3]:

#PianoRoll
sys.path.append(os.getcwd()+"/../../PianoRoll/")
sys.path.append(os.getcwd()+"/../../")
from PianoRoll import PianoRoll
from Utils import Utils
from metrics import Metrics


# In[4]:

#RNN Config
num_epochs = 10
batch_size = 128
num_layers = 3
loss_function = 'categorical_crossentropy'
optimizer = 'adam'
activ_func = "sigmoid"
validation_split = 0.1
note_threshold = 0.1
res_factor = 12
le_rate = 3e-4
dropout_W = 0.1
dropout_U = 0.1
#Composer paths
chord_dir = os.getcwd() + "/../../data/testData/chords/"
composition_dir = os.getcwd() + "/compositions/"

#Checkpoint paths
model_dir = os.getcwd() + "/checkpoints/model/"
weights_dir = os.getcwd() + "/checkpoints/weights/"
train_chord_path = os.getcwd() + "/../../data/trainData/chords/"
train_melody_path = os.getcwd() + "/../../data/trainData/melody/"
test_chord_path = os.getcwd() + "/../../data/testData/chords/"

#Results - Plots file
results_dir = os.getcwd() + "/results/"

#Other constants
MIDI_FILE_EXTENSION = "mid"
NOTE_ON = "note_on"
NOTE_OFF = "note_off"
MAX_NOTE = "max_note"
MIN_NOTE = "min_note"
TICKS = "ticks"
MIN_NOTE_INITIALIZER = 10000
MAX_NOTE_INITIALIZER = 0
MAX_TICK_INITIALIZER = 0
MELODY_LOWEST_NOTE = 60


# In[5]:

chord_roll = PianoRoll(train_chord_path, res_factor=res_factor)
melody_roll = PianoRoll(train_melody_path, res_factor=res_factor)
chord_roll_test = PianoRoll(test_chord_path, res_factor=res_factor)
print("Number of ticks:{}\tMax_Note:{}\tMin_Note:{}".format(chord_roll.ticks,
                                                            chord_roll.max_note,
                                                            chord_roll.min_note))
print("Number of ticks:{}\tMax_Note:{}\tMin_Note:{}".format(melody_roll.ticks,
                                                            melody_roll.max_note,
                                                            melody_roll.min_note))


# In[6]:

chord_data = chord_roll.generate_piano_roll_func()
mel_data = melody_roll.generate_piano_roll_func()
input_data, target_data = PianoRoll.generate_samples(chord_data, mel_data, seq_length=chord_roll.ticks)
X_train = input_data.astype(np.int32)
Y_train = target_data.astype(np.int32)


# In[7]:

D = X_train.shape[2] #Number of input dimensions
C = Y_train.shape[1] #Number of classes
model = Sequential()
reg_W = WeightRegularizer(l1=0.01, l2=0.)
reg_U = WeightRegularizer(l1=0.01, l2=0.)
reg_b = WeightRegularizer(l1=0.01, l2=0.)
if num_layers <= 1:
    model.add(LSTM(input_dim = D, 
                   output_dim=C, 
                   activation=activ_func,
                   W_regularizer = reg_W,
                   U_regularizer = reg_U,
                   b_regularizer = reg_b,
                   dropout_W = dropout_W,
                   dropout_U = dropout_U,
                   return_sequences=False))
else:
    layer_dims = int(input("Enter the number of units in hidden layer#{}\t".format(1)))
    model.add(LSTM(input_dim = D, 
                   output_dim=layer_dims, 
                   activation=activ_func,
                    W_regularizer = reg_W,
                   U_regularizer = reg_U,
                   b_regularizer = reg_b,
                   dropout_W = dropout_W,
                   dropout_U = dropout_U,
                   return_sequences=True))
    for layer_id in range(2, num_layers):
        layer_dims = int(input("Enter the number of units in the hidden layer#{}\t".format(layer_id)))
        reg_W = WeightRegularizer(l1=0.01, l2=0.)
        reg_U = WeightRegularizer(l1=0.01, l2=0.)
        reg_b = WeightRegularizer(l1=0.01, l2=0.)
        model.add(LSTM(output_dim=layer_dims, 
                       activation=activ_func,
                       W_regularizer = reg_W,
                       U_regularizer = reg_U,
                       b_regularizer = reg_b,
                       dropout_W = dropout_W,
                       dropout_U = dropout_U,
                       return_sequences=True))
    reg_W = WeightRegularizer(l1=0.01, l2=0.)
    reg_U = WeightRegularizer(l1=0.01, l2=0.)
    reg_b = WeightRegularizer(l1=0.01, l2=0.)
    model.add(LSTM(output_dim=C, 
                   activation=activ_func,
                   W_regularizer = reg_W,
                   U_regularizer = reg_U,
                   b_regularizer = reg_b,
                   dropout_W = dropout_W,
                   dropout_U = dropout_U,
                   return_sequences=False))
print(model.summary())
# print(model.get_config())


# In[8]:

if optimizer == "adam":
    opt_func = Adam(lr=le_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss=loss_function, metrics=["accuracy"], optimizer=optimizer)


# In[ ]:

history = Metrics()
model.fit(X_train, Y_train, 
          batch_size=batch_size, 
          nb_epoch=num_epochs, 
          callbacks=[history], 
          validation_split=validation_split)


# In[ ]:

weights_file = "{}layer_{}epochs_{}".format(num_layers, num_epochs, time.strftime("%Y%m%d_%H_%M.h5"))
model.save_weights(weights_dir + weights_file)
model_file = "{}layer_{}epochs_{}".format(num_layers, num_epochs, time.strftime("%Y%m%d_%H_%M.json"))
with open(model_dir + model_file, "w") as model_file_writer:
    model_file_writer.write(model.to_json())


# # Visualize the stats

# In[ ]:

results_file = "lstm_{}layer_{}epochs_{}".format(num_layers, num_epochs, time.strftime("%Y%m%d_%H_%M.jpg"))
Utils.visualize_stats(history.stats, results_file)


# In[ ]:

chord_files = glob.glob("%s*.mid" %(chord_dir))
composition_files = []
for i in range(len(chord_files)):
    composition_files.append('%d' %(i+1))
print(chord_files, composition_files)


# In[ ]:

chord_test_roll = PianoRoll(chord_dir, res_factor=res_factor)
print("# of Ticks:{}\tMax_Note:{}\tMin_Note:{}".format(chord_test_roll.ticks,
                                                  chord_test_roll.max_note,
                                                  chord_test_roll.min_note))


# In[ ]:

chord_test_data = chord_test_roll.generate_piano_roll_func()
X_test = PianoRoll.generate_test_samples(chord_test_data, chord_test_roll.ticks)


# In[ ]:

#Load Model
model_files = listdir(model_dir)
print("Choose a file for the model:")
print("---------------------------------------")
for i, file in enumerate(model_files):
    print(str(i) + " : " + file)
print("---------------------------------------")
file_index = input("Enter the index of the file model that you want:\t")
model_path = model_dir + model_files[file_index]
print("Loading model from {}".format(model_path))
model = model_from_json(open(model_path).read())


# In[ ]:

#Load Weights
weights_files = listdir(weights_dir)
print("Choose a file for the weights:")
print("---------------------------------------")
for i, file in enumerate(weights_files):
    print(str(i) + " : " + file)
print("---------------------------------------")
file_index = input("Enter the file index of the weights that you want")
weights_path = weights_dir + weights_files[file_index]
print("Loading weights from ", weights_path)
model.load_weights(weights_path)


# In[ ]:

model.compile(loss='categorical_crossentropy', optimizer='adam')
for i, song in enumerate(X_test):
    net_output = model.predict(song)
    net_roll = PianoRoll.NetOutToPianoRoll(net_output, threshold=note_threshold)
    PianoRoll.createMidiFromPianoRoll(net_roll, MELODY_LOWEST_NOTE, composition_dir,
                                               composition_files[i], note_threshold, res_factor=res_factor)
    print("Finished composing song %d." %(i+1))

