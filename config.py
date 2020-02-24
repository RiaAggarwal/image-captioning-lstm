
import torch.nn as nn

baseDataPath = './data/'
def gD(dir):
	return baseDataPath + dir

# baseResultPath = './results/'
# def gR(dir):
# 	return baseResultPath + fileName

from torchvision import transforms
# transform = transforms.Compose([ 
#         transforms.RandomCrop(args.crop_size),
#         transforms.RandomHorizontalFlip(), 
#         transforms.ToTensor(), 
#         transforms.Normalize((0.485, 0.456, 0.406), 
#                              (0.229, 0.224, 0.225))])

from architectures import EncoderCNN, DecoderLSTM

vocab_path = ''
with open(vocab_path, 'rb') as f:
	vocab = pickle.load(f)

arguments = {

	'epochs' : 1000,

	'batch_size' : 32,

	'num_workers' : 4,

	'model_path' : './results/',

	'embed_size' : 300,

	'hidden_size' : 10000,

	'val_step' : 3,

	'encoder' : EncoderCNN,

	'decoder' : DecoderLSTM,

	'vocabulary' : vocab,

	'vocab_size' : len(vocab),

	'num_layers' : 1,

	'loss_criterion' : nn.CrossEntropyLoss(),

	'learning_rate' : 0.001,

	'train_image_dir' : gD('images/train'),

	'train_json_path' : gD('annotations/captions_train2014.json'),

	'val_image_dir' : gD('images/validation'),

	'val_json_path' : gD('annotations/captions_train2014.json'),

	'test_image_dir' : gD('images/test'),

	'test_json_path' : gD('annotations/captions_val2014.json'),

	'transforms' : transform,


}