# coding=utf-8
import torch
import torch.nn as nn
from torch.autograd import Variable
from config import Config
# from main_model import BERT_LSTM_CRF
import torch.optim as optim
from utils import load_vocab, read_corpus, load_model, save_model,load_label_list
import utils
from torch.utils.data import TensorDataset
# from torch.utils.data import DataLoader
from dataloader.DataLoader import DataLoader_r as DataLoader
import fire
import numpy as np
# from tensorboardX import SummaryWriter
from calculate_F1 import *
from main_model import *
from dataloader.DataLoader import DataSetRewrite
import logging
import time


def train(**kwargs):
    config = Config()
    config.update(**kwargs)
    print('当前设置为:\n', config)
    # if config.use_cuda:
    #     torch.cuda.set_device(config.gpu)
    print('loading corpus')
    # vocab = load_vocab(config.vocab)
    label_dic= load_vocab(config.label_file)
    tagset_size = len(label_dic)
    # train_data = read_corpus(config.train_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)
    # dev_data = read_corpus(config.dev_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)

    train_dataset=DataSetRewrite(config.train_file,config.max_length,label_dic)
    dev_dataset=DataSetRewrite(config.dev_file,config.max_length,label_dic)

    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=config.batch_size, shuffle=True)

    model = BERT_LSTM_CRF(config.bert_path, tagset_size, config.bert_embedding, config.rnn_hidden, config.rnn_layer, dropout_ratio=config.dropout_ratio, dropout1=config.dropout1, use_cuda=config.use_cuda)
    if config.load_model:
        assert config.load_path is None
        model = load_model(model, name=config.load_path)
    model.to(device)
    # dev(model, dev_loader, 10, config)
    # need_frozen_list=['word_embeds']
    # optimizer=optim.Adam()
    optimizer = getattr(optim, config.optim)
    for parm in model.named_parameters():
        if 'word_embeds' == parm[0][:11]:
            parm[1].requires_grad=False
    # for parm1 in model.named_parameters():
    #   print(parm1[0][:12])
    #   print(parm1,parm1[1].requires_grad)
    optimizer = optimizer(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr, weight_decay=config.weight_decay)
    eval_loss = 10000
    architect = Architect(model, config)
    for epoch in range(config.base_epoch):
        step = 0
        train_start_time=time.time()
        logging.info('-------------------------------start {} epoch train-----------------------------------'.format(epoch))
        model.train()
        for batch in train_loader:
            step += 1
            model.zero_grad()
            inputs, masks ,tags = batch
            feats = model(inputs)
            for i,tag in enumerate(tags):
                tags[i]=torch.cat([tag,torch.zeros(model.batch_sequence_length-len(tag))])
                masks[i]=torch.cat([masks[i],torch.zeros(model.batch_sequence_length-len(tag))])
            tags=Variable(torch.cat(tags).long()).view(config.batch_size,model.batch_sequence_length)
            masks =Variable(torch.cat(masks).long()).view(config.batch_size,model.batch_sequence_length)

            # print(torch.cuda.get_device_name(),'qwerqrtqert',torch.cuda.current_device())

            tags=tags.to(device)
            masks=masks.to(device)
            # print(tags.shape,tags.device,'asdf',device)
            loss = model.loss(feats, masks,tags)
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                print('step: {} |  epoch: {}|  loss: {}'.format(step, epoch, loss.item()))
        arch_train(dev_loader,architect)

        logging.info('End {} epoch train,spend time {}.\nThe arch_parmeters is {},the top 3 parameters is {} '.format(epoch,time.time()-train_start_time,model.arch_parameters(),torch.topk(model.arch_parameters(),3)))
        loss_temp = test_mod(model, dev_loader, epoch, config)
        # print(loss_temp)
        if loss_temp < eval_loss:
            save_model(model, epoch)
        eval_loss=loss_temp

def arch_train(dev_loader,architect):
    try:
        input_search, target_search ,masks_search= next(dev_loader)
    except:
        valid_queue_iter = iter(dev_loader)
        input_search, target_search,masks_search = next(valid_queue_iter)
    architect.step(input_search,target_search,masks_search)


def test_mod(model, dev_loader, epoch,config):
    start_dev_time=time.time()
    logging.info('Start test')
    model.eval()
    eval_loss = 0
    true = []
    pred = []
    length = 0
    entity_list_1=[]
    with open(config.entity_list) as txt:
        for line in txt:
            entity_list_1.append(line.strip())
    predict_correct = dict(zip(entity_list_1,[0]*len(entity_list_1)))
    predict_total_enti=dict(zip(entity_list_1,[0]*len(entity_list_1)))
    total_enti_num=dict(zip(entity_list_1,[0]*len(entity_list_1)))
    target_list = {}
    for batch in dev_loader:
        inputs, masks, tags = batch
        length += len(masks)
        feats = model(inputs, masks)
        tags,masks=tag_mask_padding(model,masks,tags,config)
        path_score, best_path = model.crf(feats, masks.byte())
        # best_path=inputs
        counter=Metrics(entity_exchang(tags,config), entity_exchang(best_path,config))
        counter.compared_tag(inputs,best_path,tags)
        for entity in counter.entity_set:
            if entity in counter.corrent_entity_number:
                predict_correct[entity]+=counter.corrent_entity_number[entity]
            if entity in counter.predict_entity_counter:
                predict_total_enti[entity]+=counter.predict_entity_counter[entity]
            total_enti_num[entity]+=counter.std_entity_counter[entity]
        loss = model.loss(feats, masks, tags)
        eval_loss += loss.item()
        pred.extend([t for t in best_path])
        true.extend([t for t in tags])
    entity_F1=dict(zip(entity_list_1,[0]*len(entity_list_1)))
    precision={}
    recall={}
    for entity in entity_list_1:
        precision[entity]=predict_correct[entity]/max(1e-10,predict_total_enti[entity])
        recall[entity]=predict_correct[entity]/max(1e-10,total_enti_num[entity])
        entity_F1[entity]=2*(precision[entity]*recall[entity])/max(1e-10,(precision[entity]+recall[entity]))
    logging.info('End dev,spend time {}.\n F1 value is {} ,eval  epoch: {}|  loss: {}.'.format(time.time()-start_dev_time,entity_F1,epoch, eval_loss/length))
    logging.info('-----------------------------End {} epoch train---------------------------------'.format(epoch))
    print('F1 value is {}'.format(entity_F1))
    print('eval  epoch: {}|  loss: {}'.format(epoch, eval_loss/length))
    model.train()
    return eval_loss

def entity_exchang(sequnce_list,config):
    result=[]
    label_list=load_label_list(config.label_file)
    for sequnce in sequnce_list:
        sentence_stride=0
        sentence_list=[]
        flag = 0
        while sentence_stride < len(sequnce):

            str_label=label_list[sequnce[sentence_stride]]
            sentence_list.append(str_label)
            if str_label[:2]=='B-':
                flag=1
                if sentence_stride==len(sequnce)-1 or label_list[sequnce[sentence_stride+1]][:2]!='I-':
                    sentence_list[-1] = 'S-' + str_label[2:]
                    flag=0
                    sentence_stride += 1
                    continue
            if flag==1:
                if sentence_stride==len(sequnce)-1 or label_list[sequnce[sentence_stride+1]][:2]!='I-':
                    sentence_list[-1] = 'E-' + str_label[2:]
                    flag = 0
                    sentence_stride += 1
                    continue
                # if label_list_1[sequnce[sentence_stride+1]][:2]!='I-':
                #     # if str_label[:2]=='B-':
                #     #     sentence_list[-1]='S-'+str_label[2:]
                #     #     flag=0
                #     # else:
                #     sentence_list[-1]='E-'+str_label[2:]
                #     flag=0
                #     continue
                elif str_label[:2]=='I-':
                    sentence_list[-1]='M-'+str_label[2:]
            sentence_stride+=1
        result.append(sentence_list)
    return result

'''以下为测试用代码'''
# config = Config()
# # label_list_1 = load_label_list(config.label_file)
# label_list_1=[]
# with open(config.label_file) as w:
#     for label in w:
#         label_list_1.append(label)
# config=Config()
# label_dic=load_vocab(config.label_file)
# vocab=load_vocab(config.vocab)
# dev_data = read_corpus(config.dev_file, max_length=config.max_length, label_dic=label_dic, vocab=vocab)
# dev_ids = torch.LongTensor([temp.input_id for temp in dev_data])
# dev_masks = torch.LongTensor([temp.input_mask for temp in dev_data])
# dev_tags = torch.LongTensor([temp.label_id for temp in dev_data])
# dev_dataset = TensorDataset(dev_ids, dev_masks, dev_tags)
# dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=config.batch_size)
# label_dic= load_vocab(config.label_file)
# tagset_size = len(label_dic)
# model = BERT_LSTM_CRF(config.bert_path, tagset_size, config.bert_embedding, config.rnn_hidden, config.rnn_layer, dropout_ratio=config.dropout_ratio, dropout1=config.dropout1, use_cuda=config.use_cuda)
# # model = load_model(model, name=config.load_path)
# model.eval()
'''以下注释为测试用代码'''
# test_sequnce_p='O O O O O O O O O O O B-PRAC I-PRAC O O O O O O O O O O B-PRAC O B-PRAC O B-PRAC I-PRAC I-PRAC O O O O '
#
# test_sequnce_t='O O O O O O O O O O O B-PRAC I-PRAC I-PRAC O O O O O O O O O B-PRAC I-PRAC I-PRAC O B-PRAC I-PRAC I-PRAC O O O O '
#
# p_list=[]
# t_list=[]
# for i in range(len(test_sequnce_p.split())):
#     p_list.append(label_dic[test_sequnce_p.split()[i]])
#     t_list.append(label_dic[test_sequnce_t.split()[i]])
# dev_loader = DataLoader(TensorDataset(torch.tensor([p_list]),torch.randint(10,[1,len(p_list)]),torch.tensor([t_list])), shuffle=True, batch_size=config.batch_size)

# config=Config()
# for i, batch in enumerate(dev_loader):
#     inputs, masks, tags = batch
#     dev(model,dev_loader,100,config)
#     print('epoch'.format(i))

def predict(text_file):
    config=Config()
    label_dic = load_vocab(config.label_file)
    tagset_size = len(label_dic)
    model = BERT_LSTM_CRF(config.bert_path, tagset_size, config.bert_embedding, config.rnn_hidden, config.rnn_layer,
                          dropout_ratio=config.dropout_ratio, dropout1=0.5, use_cuda=config.use_cuda)
    model = load_model(model, name=config.load_path)

    model.eval()
    vocab = load_vocab(config.vocab)
    with open(text_file,encoding='utf-8') as file:
        for text in file:
            test_data=utils.load_test(text,max_length=config.max_length,vocab=vocab)
            test_ids = torch.LongTensor([temp.input_id for temp in [test_data]])
            test_masks = torch.LongTensor([temp.input_mask for temp in [test_data]])
            # with SummaryWriter(comment='LeNet') as w:
            #     w.add_graph(model,test_ids)
            feats = model(test_ids, test_masks)
            path_score, best_path = model.crf(feats, test_masks.byte())
            list_1=best_path[:len(text)]
            print('输入文本：{}'.format(text))
            for j, k in zip(text, list_1[1:-1]):
                print(j, k, end='\n')
            print(text, '\n', list_1[1:-1])
            print(path_score, best_path)
            std_label=best_path[:]

def tag_mask_padding(model,masks,tags,config):
    for i, tag in enumerate(tags):
        tags[i] = torch.cat([tag, torch.zeros(model.batch_sequence_length - len(tag))])
        masks[i] = torch.cat([masks[i], torch.zeros(model.batch_sequence_length - len(tag))])
    tags = Variable(torch.cat(tags).long()).view(config.batch_size, model.batch_sequence_length)
    masks = Variable(torch.cat(masks).long()).view(config.batch_size, model.batch_sequence_length)
    tags = tags.to(device)
    masks = masks.to(device)
    return tags,masks

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(message)s',
                        filename='time_recoder.log',
                        filemode='a',
                        level=logging.INFO)
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    fire.Fire()

    # test_data='./data2/test.txt'
    # predict(test_data)
