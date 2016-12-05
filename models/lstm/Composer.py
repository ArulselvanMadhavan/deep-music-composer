
# coding: utf-8

# In[2]:

import numpy as np
import glob
import os
from os import listdir
import sys
# get_ipython().magic(u'load_ext autoreload')
# get_ipython().magic(u'autoreload 2')


# In[3]:

#PianoRoll
sys.path.append(os.getcwd()+"/../../PianoRoll/")
sys.path.append(os.getcwd()+"/../../")
from PianoRoll import PianoRoll
import Constants


# In[4]:

#Keras Imports
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers.recurrent import LSTM


# In[6]:

#Composer paths
chord_dir = os.getcwd() + "/../../data/testData/chords/"
composition_dir = os.getcwd() + "/compositions/"


# In[7]:

chord_files = glob.glob("%s*.mid" %(chord_dir))


# In[8]:

composition_files = []
for i in range(len(chord_files)):
    composition_files.append('%d' %(i+1))
print(chord_files, composition_files)


# ## User Input Configs

# In[9]:

#Composition Configurations
res_factor = 12
#Piano Roll Threshold
threshold = 0.1


# In[10]:

#TODO Remove the dependencies
import data_utils_compose


# In[11]:

chord_test_roll = PianoRoll(chord_dir, res_factor=res_factor)
print("# of Ticks:{}\tMax_Note:{}\tMin_Note:{}".format(chord_test_roll.ticks,
                                                  chord_test_roll.max_note,
                                                  chord_test_roll.min_note))


# In[12]:

chord_test_data = chord_test_roll.generate_piano_roll_func()
X_test = PianoRoll.generate_test_samples(chord_test_data, chord_test_roll.ticks)


# In[14]:

#Load Model
model_dir = os.getcwd() + "/checkpoints/model/"
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


# In[16]:

#Load Weights
weights_dir = os.getcwd() + "/checkpoints/weights/"
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


# ## TODO: Find a way to load the constants used below

# In[25]:

model.compile(loss='categorical_crossentropy', optimizer='adam')
for i, song in enumerate(X_test):
    net_output = model.predict(song)
    net_roll = PianoRoll.NetOutToPianoRoll(net_output, threshold=threshold)
    PianoRoll.createMidiFromPianoRoll(net_roll, Constants.MELODY_LOWEST_NOTE, composition_dir,
                                               composition_files[i], threshold, res_factor=res_factor)
    print("Finished composing song %d." %(i+1))


# # Visualize the Notes in Sheet Notation

# In[27]:

from unroll import midi2keystrikes
# keystrikes = midi2keystrikes("../../data/trainData/chords/001.mid")
# keystrikes.quarter_duration = [50,100,0.02]
# keystrikes.transcribe('score.ly', quarter_durations=[50, 100, 0.02])


# In[ ]:




# In[18]:

#Test Roll
chord_lowest_note, chord_highest_note, chord_ticks = data_utils_compose.getNoteRangeAndTicks(chord_files, 
                                                                                             res_factor=res_factor)
chord_roll = data_utils_compose.fromMidiCreatePianoRoll(chord_files, 
                                                        chord_ticks, 
                                                        chord_lowest_note,
                                                        res_factor=res_factor)
double_chord_roll = data_utils_compose.doubleRoll(chord_roll)

test_data = data_utils_compose.createNetInputs(double_chord_roll, seq_length=chord_ticks)
np.testing.assert_equal(X_test, test_data)


# In[ ]:




# In[ ]:



