import torch
a=[torch.ones([3]) for _  in range(4)]

c=torch.tensor(a)

# c=[sen for sen in a]
# print(c)
# # b=[torch.tensor(range(12))]
# # c=a+b
# d=torch.cat(c)
# # print(d)