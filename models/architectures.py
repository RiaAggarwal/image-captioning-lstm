import torch
import torch.nn as nn
import torchvision.models as pretrained
import numpy as np
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler


resNetComplete = pretrained.resnext50_32x4d(pretrained=True)

class EncoderCNN(nn.Module):
	def __init__(self, output_size):
		super(EncoderCNN, self).__init__()
		resNetComplete = pretrained.resnext50_32x4d(pretrained=True)
		subModules = list(resNetComplete.children())[:-1]
		self.resNetToUse = nn.Sequential(*subModules)
		self.lastLayer = nn.Linear(resNetComplete.fc.in_features, output_size)
		self.batchNorm = nn.BatchNorm1d(output_size)

	def forward(self, image_inputs):
		imageFeatures = self.resNetToUse(image_inputs)
		imageFeatures = imageFeatures.reshape(imageFeatures.size(0), -1)
		outputFeatures = self.batchNorm(self.lastLayer(imageFeatures))
		return outputFeatures

class DecoderLSTM(nn.Module):
	def __init__(self, input_size, hidden_size, vocab_size, num_layers, max_sentence_length = 20):
		super(DecoderLSTM, self).__init__()
		self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
		self.outputLayer = nn.Linear(hidden_size, vocab_size)
		self.embeddingLayer = nn.Embedding(vocab_size, input_size)
		self.maxSentenceLength = max_sentence_length

	def forward(self, encoder_outputs, captions, lengths):
		wordEmbeddings = self.embeddingLayer(captions)
		wordEmbeddings = torch.cat((encoder_outputs.unsqueeze(1), wordEmbeddings), 1)
		hiddenStates, _ = self.lstm(torch.nn.utils.rnn.pack_padded_sequence(wordEmbeddings, lengths, batch_first=True))
		vocabScores = self.outputLayer(hiddenStates[0])
		return vocabScores

	def generate(self, features, states=None):
		word_ids = []
		features = features.unsqueeze(1)
		for i in range(self.maxSentenceLength):
			hiddens, states = self.lstm(features, states)
			outputs = self.outputLayer(hiddens.squeeze(1))
			_ , predicted = outputs.max(1)
			word_ids.append(predicted)
			inputs = self.embeddingLayer(predicted)
			inputs = inputs.unsqueeze(1)
		word_ids = torch.stack(word_ids, 1)
		return word_ids
    
    def generate_captions(self, logits, mode='deterministic', t=1):
        if (mode == 'deterministic'):
            _, predicted = logits.max(1)
            word_id = predicted
        
        elif(mode == 'stochastic'):
            soft_out = F.softmax(logits/t, dim=1)
            word_id = WeightedRandomSampler(torch.squeeze(soft_out), 1) #get only one sample. change it to get more samples
        
        return word_id
            
            
            
            
            
        

# e = EncoderCNN(300)
# output = e(torch.zeros(3, 3, 224, 224))
# d = DecoderLSTM(300, 200, 50, 1, 5)
# o = d(output, torch.from_numpy(np.array([[1,2,1,3,3],[1,2,1,3,3],[1,2,1,3,3]])), torch.from_numpy(np.array([3,2,1])))








