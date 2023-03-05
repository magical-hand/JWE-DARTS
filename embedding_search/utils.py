# coding=utf-8
import random

import torch
import os
import datetime
import unicodedata


class InputFeatures(object):
    def __init__(self, input_id, label_id, input_mask):
        self.input_id = input_id
        self.label_id = label_id
        self.input_mask = input_mask

def load_label_list(vocab_file):
    key_list=[]
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            # token = token.strip()
            # if token not in ['<pad>','O','<start>','<eos>'] and token[:2]!='I-':
            key_list.append(token)
    return key_list

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = {}
    key_list=[]
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            key_list.append(token)
            vocab[token] = index
            index += 1
    return vocab

def load_test(text1,max_length,vocab):
    # tokens = text1.split()
    tokens=list(text1)
    if len(tokens) > max_length-2:
        tokens = tokens[0:(max_length-2)]
    tokens_f = ['[CLS]'] + tokens + ['[SEP]']
    input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in tokens_f]
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_length:
        input_ids.append(0)
        input_mask.append(0)
    assert len(input_ids) == max_length
    assert len(input_mask) == max_length
    feature = InputFeatures(input_id=input_ids, input_mask=input_mask, label_id=None)
    return feature

def read_corpus(path, max_length, label_dic, vocab):
    """
    :param path:数据文件路径
    :param max_length: 最大长度
    :param label_dic: 标签字典
    :return:
    """
    file = open(path, encoding='utf-8')
    content = file.readlines()
    file.close()
    result = [[],[],[]]
    for line in content:
        text, label = line.strip().split('|||')
        tokens = text.split()
        label = label.split()
        if len(tokens) > max_length-2:
            tokens = tokens[0:(max_length-2)]
            label = label[0:(max_length-2)]
        tokens_f =['[CLS]'] + tokens + ['[SEP]']
        label_f = ["<start>"] + label + ['<eos>']
        # input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in tokens_f]
        label_ids = [label_dic[i] for i in label_f]
        input_mask = [1] * len(tokens_f)
        # while len(tokens_f) < max_length:
        #     tokens_f.append('<pad>')
        #     input_mask.append(0)
        #     label_ids.append(label_dic['<pad>'])
        label_ids=torch.LongTensor(label_ids)
        input_mask=torch.LongTensor(input_mask)
        result[0].append(tokens_f)
        result[1].append(input_mask)
        result[2].append(label_ids)
    return result


def save_model(model, epoch, path='result', **kwargs):
    """
    默认保留所有模型
    :param model: 模型
    :param path: 保存路径
    :param loss: 校验损失
    :param last_loss: 最佳epoch损失
    :param kwargs: every_epoch or best_epoch
    :return:
    """
    if not os.path.exists(path):
        os.mkdir(path)
    if kwargs.get('name', None) is None:
        cur_time = datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')
        name = cur_time + '--epoch:{}'.format(epoch)
        full_name = os.path.join(path, name)
        # print(list(model.named_parameters()))
        torch.save(model.state_dict(), full_name)
        print('Saved model at epoch {} successfully'.format(epoch))
        with open('{}/checkpoint'.format(path), 'w') as file:
            file.write(name)
            print('Write to checkpoint')


def load_model(model, path='result', **kwargs):
    if kwargs.get('name', None) is None:
        with open('{}/checkpoint'.format(path)) as file:
            content = file.read().strip()
            name = os.path.join(path, content)
    else:
        name=kwargs['name']
        name = os.path.join(path,name)
    model.load_state_dict(torch.load(name, map_location=lambda storage, loc: storage))
    print('load model {} successfully'.format(name))
    return model

class data_loader_1(object):
    def __init__(self,data,batch_size):
        self.data_set=data[0]
        self.label=data[2]
        self.mask=data[1]
        self.batch_size=batch_size

    def __iter__(self):
        self.a=0
        self.item_num = len(self.data_set)
        self.iter_list = random.sample(range(0, self.item_num), self.item_num)
        # print(self.iter_list,">>>>>")
        return self

    def __next__(self):
        if self.a == self.item_num//self.batch_size:
            raise StopIteration
        data_set=[self.data_set[i] for i in self.iter_list[self.a*self.batch_size:(self.a+1)*self.batch_size]]
        mask=[self.mask[i] for i in self.iter_list[self.a*self.batch_size:(self.a+1)*self.batch_size]]
        lable=[self.label[i] for i in self.iter_list[self.a*self.batch_size:(self.a+1)*self.batch_size]]
        self.a+=1
        return data_set,mask,lable
