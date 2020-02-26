import torch.nn as nn
import pickle
baseDataPath = './data/'
def gD(dir):
	return baseDataPath + dir

def get_ids(file_path):
    ids = []
    with open(file_path, 'r') as file:
        str_ids = file.read().split(",")
        ids = [int(_ids) for _ids in str_ids]
    return ids

# baseResultPath = './results/'
# def gR(dir):
# 	return baseResultPath + fileName

from torchvision import transforms

size = (224,224) # refer to data_loader_captions notebook
transforms_ = transforms.Compose([
                    transforms.CenterCrop(size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

train_ids = get_ids('./data/train_ids.txt')
val_ids = get_ids('./data/val_ids.txt')
test_ids = get_ids('./data/test_ids.txt')

from models.architectures import EncoderCNN, DecoderLSTM

vocab = ''
vocab_path = './vocab/word_to_idx.p'
with open(vocab_path, 'rb') as f:
	vocab = pickle.load(f)

arguments = {

	'epochs' : 1000,

	'batch_size' : 2,

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

	'train_image_ids' : train_ids,

	'train_json_path' : gD('annotations/captions_train2014.json'),

	'val_image_ids' : val_ids,

	'val_json_path' : gD('annotations/captions_train2014.json'),

	'test_image_ids' : test_ids,

	'test_json_path' : gD('annotations/captions_val2014.json'),

	'transforms' : transforms_,

	'root' : '/datasets/COCO-2015/train2014/'

}