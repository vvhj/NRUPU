import os
import random
import numpy as np
from constant import *
import json
from tqdm import tqdm
from PIL import Image
row_anchor = yq_row_anchor
data_root = '/root/file/yqfod/imagesv2'
list_path = os.path.join(data_root, 'test_gt.txt')
names = os.listdir(data_root)
fp = open(list_path,'w')
fl = []
for name in tqdm(names):
    labelpath = "imagesv2/"+name
    fp.write(labelpath+'\n')
fp.close()
