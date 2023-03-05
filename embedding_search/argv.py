import sys
# import argparse
# for i in sys.argv:
#       print(i)
# print(sys.argv[0])
#
# parser=argparse.ArgumentParser('智能搜救')
# parser.add_argument('--gpu_nums',type=int,help='number of gpu',default=0)
# argv=parser.parse_args()

import numpy as np

a=np.load('C:\\Users\\咸鱼\\.flair\\embeddings\\zh-wiki-fasttext-300d-1M.vectors.npy')
b=np.load('C:\\Users\\咸鱼\\.flair\\embeddings\\zh-wiki-fasttext-300d-1M')
print(a)