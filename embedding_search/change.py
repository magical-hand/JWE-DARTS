# coding=utf-8
from config import *

def read_tsv(file_path, split_seg=" "):
    """
    read tsv style data1
    :param file_path: file path
    :param split_seg: seg
    :return: [(sentence, label), ...]
    """
    total_data=[]
    data = []
    sentence = []
    label = []
    line_num = 0
    line_level=0
    config=Config()
    label_tags=[]
    with open(config.label_file) as w:
        for line in w:
            label_tags.append(line.strip())
    with open(file_path, 'r', encoding='utf-8') as reader:
        for line in reader:
            line_level += 1
            line=line.strip()
            if not line:
                data.append((sentence, label))
                sentence = []
                label = []
                line_num = 0
            if line_num == Config().max_length - 1:
                if sentence:
                    pos=0
                    while label[line_num-pos-1]!= 'O':
                        pos+=1
                    data.append((sentence[:len(sentence)-pos-1], label[:len(sentence)-pos-1]))
                    sentence = sentence[-pos-1:]
                    label = label[-pos-1:]
                    line_num=pos+1
                continue
            if split_seg not in line and line:
                print('label {} format false'.format(line))
            line=line.split(split_seg)
            if line[-1] not in label_tags and line[-1]:
                print('error tag {}'.format(line_level))
                continue
            if len(line)!=2:
                print('dellet line {}'.format(line_level))
                continue
            sentence.append(line[0])
            label.append(line[-1])
            line_num+=1
    if sentence:
        data.append((sentence, label))
    return data
def write_file(source_file_path,file_path):
    Note=open(file_path, mode='w+', encoding='utf-8')
    # write_data=read_tsv("data2/train1_t.txt")
    write_data=read_tsv(source_file_path)

    text_length=Config().max_length-2
    pos=0
    for data1 in write_data:
        if len(data1[0])!=len(data1[1]):
            print('program error')
        for data2 in data1[0]:
            Note.write(data2+' ')
        Note.write('|||')
        for data3 in data1[1]:
            Note.write(data3+' ')
        Note.write('\n')
    # for data1 in write_data_2:
    #     for data2 in data1[0]:
    #         Note.write(data2+' ')
    #     Note.write('|||')
    #     for data3 in data1[1]:
    #         Note.write(data3+' ')
    #     Note.write('\n')
    Note.close()

main_path='./CLUENER2020/'
sor_char="_t"
source_train_file='train{}.txt'.format(sor_char)
source_dev_file='dev{}.txt'.format(sor_char)
sorce_test_file='test{}.txt'.format(sor_char)

sor_char=''
write_train_file='train{}.txt'.format(sor_char)
write_dev_file='dev{}.txt'.format(sor_char)
write_test_file='test{}.txt'.format(sor_char)

write_file(main_path+source_train_file,main_path+write_train_file)
write_file(main_path+source_dev_file,main_path+write_dev_file)
write_file(main_path+sorce_test_file,main_path+write_test_file)
