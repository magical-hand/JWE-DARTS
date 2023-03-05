from torch.utils.data import Dataset
import torch
import random

class DataSetRewrite(Dataset):

    def __init__(self,data_path,max_length,label_dic):
        super(DataSetRewrite, self).__init__()

        self.data_path='./'+data_path

        file = open(self.data_path, encoding='utf-8')
        content = file.readlines()
        file.close()
        self.data_set=[]
        self.labels_set=[]
        self.masks_set=[]
        for line in content:
            text, label = line.strip().split('|||')
            tokens = text.split()
            label = label.split()
            if len(tokens) > max_length - 2:
                tokens = tokens[0:(max_length - 2)]
                label = label[0:(max_length - 2)]
            tokens_f = tokens
            label_f = ["<start>"] + label + ['<eos>']
            # input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in tokens_f]
            label_ids = [label_dic[i] for i in label_f]
            input_mask = [1] * (len(tokens_f)+2)
            label_ids = torch.LongTensor(label_ids)
            input_mask = torch.LongTensor(input_mask)
            self.data_set.append(tokens_f)
            self.labels_set.append(input_mask)
            self.masks_set.append(label_ids)


    def __getitem__(self, index):

        return self.data_set[index],self.labels_set[index],self.masks_set[index]

    def __len__(self):
        return len(self.data_set)

class DataLoader_r():
    def __init__(self,dataset,batch_size,shuffle=True):
        self.batch_size=batch_size
        self.dataset=dataset
        self.shuffle=shuffle
        if self.shuffle==False:
            self.fetch_list=[range(0,len(self.dataset))]

    def __iter__(self):
        if self.shuffle==True:
            self.fetch_list=random.sample(range(0, len(self.dataset)), len(self.dataset))
        self.step=0
        return self

    def __next__(self):
        if self.step<len(self.dataset)//self.batch_size:

            sample_list=self.fetch_list[self.step*self.batch_size:(self.step+1)*self.batch_size]
            self.step += 1
            return [[self.dataset[i][j] for i in sample_list] for j in range(3)]
        else:
            raise StopIteration

