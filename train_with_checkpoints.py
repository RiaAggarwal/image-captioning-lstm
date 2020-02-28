#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nntools as nt


# In[2]:


from models.architectures import *
import torch
import torch.nn as nn
from config import args
import os
from data_loader import get_loader
from torch.nn.utils.rnn import pack_padded_sequence


# In[3]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[4]:


encoder = args['encoder'](args['embed_size']).to(device)
decoder = args['decoder'](args['embed_size'], args['hidden_size'], args['vocab_size'], args['num_layers']).to(device)


# In[5]:


criterion = args['loss_criterion']


# In[6]:


params = list(list(encoder.parameters()) + list(decoder.parameters()))


# In[7]:


#optimizer = torch.optim.SGD(params, lr=arguments['learning_rate'])

# Observe that all parameters are being optimized
optimizer = torch.optim.Adam(params, lr=args['learning_rate'],betas=(args['beta'], 0.999))

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


# In[8]:


stats_manager = nt.StatsManager()


# In[9]:


exp1 = nt.Experiment(encoder, decoder, device, criterion, optimizer, stats_manager, 
                     output_dir="test_3", perform_validation_during_training=True)


# In[10]:


exp1.run(num_epochs=args["epochs"])


# In[ ]:




