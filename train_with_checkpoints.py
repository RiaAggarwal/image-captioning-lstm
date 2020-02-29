#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nntools as nt


# In[2]:


from models.architectures import *
import torch
import torch.nn as nn
from config import arguments
import os
from data_loader import get_loader
from torch.nn.utils.rnn import pack_padded_sequence


# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[4]:


encoder = arguments['encoder'](arguments['embed_size']).to(device)
decoder = arguments['decoder'](arguments['embed_size'], arguments['hidden_size'], arguments['vocab_size'], arguments['num_layers']).to(device)


# In[5]:


criterion = arguments['loss_criterion']


# In[6]:


params = list(list(encoder.parameters()) + list(decoder.parameters()))


# In[7]:


optimizer = torch.optim.Adam(params, lr=arguments['learning_rate'])


# In[8]:


stats_manager = nt.StatsManager()


# In[9]:


exp1 = nt.Experiment(encoder, decoder, device, criterion, optimizer, stats_manager, 
                     output_dir=arguments['model_path'], perform_validation_during_training=True)


# In[10]:


exp1.run(num_epochs=arguments['epochs'])


# In[ ]:




