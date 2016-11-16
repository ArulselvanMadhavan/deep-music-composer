
# coding: utf-8

# In[10]:

import numpy as np
# get_ipython().magic(u'load_ext autoreload')
# get_ipython().magic(u'autoreload 2')


# In[88]:

train_data_path = "./playData/chords/"
resolution_factor = 12


# In[89]:

from PianoRoll import PianoRoll


# In[92]:

train_obj = PianoRoll(train_data_path, res_factor=24)


# In[93]:

print(train_obj.ticks, train_obj.max_note, train_obj.min_note)


# In[95]:

train_obj.generate_piano_roll_func()


# In[ ]:



