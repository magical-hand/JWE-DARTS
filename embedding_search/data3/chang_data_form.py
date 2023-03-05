import numpy as np


def change_data_form_code():
    # data_sourse='train.txt'
    # data_changed='train_t.txt'

    # data_sourse='valid.txt'
    # data_changed='valid.txt'

    data_sourse='test.txt'
    data_changed='test_t.txt'
    with open(data_changed,'w+') as w:
        with open(data_sourse) as f:
            step=0
            write_txt=''
            write_label=''
            for line in f:
                step+=1
                line=line.strip()
                if line == '':
                    if write_txt!='':
                        w.write(write_txt+'|||'+write_label+'\n')
                        write_txt=''
                        write_label=''
                    continue
                line_data=line.strip().split(' ')
                if len(line_data)==4:
                    if write_txt!='':
                        write_txt=write_txt+' '+line_data[0]
                        write_label=write_label+' '+line_data[3]
                        print(step)
                    else:
                        write_txt=line_data[0]
                        write_label=line_data[3]
                else:
                    print('data form wrong in {} line'.format(step))

def spilt_traindata():
    with open('train_t1.txt','w') as w1, open('valid_t.txt','w') as w2:
        with open('train_t.txt') as f:
            step=0
            line_nums=14986
            valid_list=np.random.randint(0,line_nums,line_nums//3)
            for line in f:
                if step in valid_list:
                    w2.write(line)
                else:
                    w1.write(line)
                step+=1

if __name__=='__main__':
    spilt_traindata()






