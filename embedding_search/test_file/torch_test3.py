from torch import nn
from flair.data import Sentence
from flair.embeddings import WordEmbeddings, FlairEmbeddings ,StackedEmbeddings
import torch
from flair import device

class victor(nn.Module):
  def __init__(self,use_cuda):
    super(victor,self).__init__()
    # self.embeddings=stackembed()
    self.embeddings = StackedEmbeddings([WordEmbeddings('glove'), FlairEmbeddings('news-forward')])
    # if use_cuda:
    #     self.embeddings.cuda()
    # self.embeddings.to(device)

  def forward(self,x):
      # return  self.embeddings(x)
    F=x
    self.embeddings.embed(F)
    print('why?')
    result=torch.cat([torch.cat([i for i in j._embeddings.values()]) for j in F.tokens])
    return result


class stackembed(nn.Module):
  def __init__(self):
    super(stackembed,self).__init__()
    embeddings = StackedEmbeddings([WordEmbeddings('glove'), FlairEmbeddings('news-forward')])
    self.embeddings=embeddings

  def forward(self,x):
    F=x
    self.embeddings.embed(F)
    print('why?')
    result=torch.cat([torch.cat([i for i in j._embeddings.values()]) for j in F.tokens])
    return result

# torch.cuda.set_device('cuda:0')

model=victor(use_cuda=True)
model.to(device)
# model.cuda()
# while True:
sentence='i am a salt fish'
# while True:
sentence=Sentence(sentence)
output=model(sentence)
# embeddings = StackedEmbeddings([WordEmbeddings('glove'), FlairEmbeddings('news-forward')])
# while True:
#   sentence='i am a salt fish'
#   sentence=Sentence(sentence)
#   embeddings.embed(sentence)

