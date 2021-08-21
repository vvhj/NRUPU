import os
import random
import numpy as np
from constant import *
import json
from PIL import Image
row_anchor = yq_row_anchor
data_root = '/root/file/yqfod'
list_path = os.path.join(data_root, 'train_gt.txt')
train_list_path = os.path.join(data_root, 'train_gt_train.txt')
val_list_path = os.path.join(data_root, 'train_gt_val.txt')
test_path = os.path.join(data_root, 'test.txt')
label_path = os.path.join(data_root,'label.json')
val_label_path = os.path.join(data_root,'test_label.json')
with open(list_path, 'r') as f:
    list1 = f.readlines()
#index = random.sample(range(0,len(list1)),int(len(list1)*0.9))
#vinde = list(set(list(range(len(list1))))-set(index))
names = [n.split(" ")[0] for n in list1]
labels = [n.split(" ")[1] for n in list1]

scale_f = lambda x : int((x * 1.0/288) * 720)

h_samples = list(map(scale_f,row_anchor))
lanes = []
names_val = names
labels_val = labels
usefull = []
fp = open(label_path,'w')
for i,name in enumerate(names_val):
    labelpath = os.path.join(data_root, labels_val[i])
    label = Image.open(labelpath)
    #label = np.array(label)
    tmp_dict = {}
    lanes = []
    print("{}  {}".format(i,name))
    uf = True
    for lane_idx in range(1,5):
        lane = []
        for r in h_samples:
            label_r = np.asarray(label)[int(round(r))]
            pos = np.where(label_r==lane_idx)[0]
            if len(pos) == 0:
                lane.append(-2)
                continue
            pos = np.mean(pos)
            lane.append(int(pos))
        if np.mean(lane) == -2:
            uf = False
        lanes.append(lane)
    if uf:
        usefull.append(i)
    tmp_dict['lanes'] = lanes
    tmp_dict['h_samples'] = h_samples
    tmp_dict['raw_file'] = name
    tmp_dict['run_time'] = 10
    
    json_str = json.dumps(tmp_dict)
    fp.write(json_str+'\n')
fp.close()

list_train = []#list1[index]
list_val = []#list1[vinde]
names_val = []
labels_val = []
for i in range(len(list1)):
    if i not in usefull:
        continue
    if i %10 == 3 or i % 10 == 7:
        list_val.append(list1[i])
        names_val.append(names[i])
        labels_val.append(labels[i])
    else:
        list_train.append(list1[i])
with open(train_list_path, 'w') as f:
    f.writelines(list_train)

with open(val_list_path, 'w') as f:
    f.writelines(list_val)
test_v = [i+'\n' for i in names_val]
with open(test_path, 'w') as f:
    f.writelines(test_v)
fp = open(val_label_path,'w')
for i,name in enumerate(names_val):
    labelpath = os.path.join(data_root, labels_val[i])
    label = Image.open(labelpath)
    #label = np.array(label)
    tmp_dict = {}
    lanes = []
    print("{}  {}".format(i,name))
    for lane_idx in range(1,5):
        lane = []
        for r in h_samples:
            label_r = np.asarray(label)[int(round(r))]
            pos = np.where(label_r==lane_idx)[0]
            if len(pos) == 0:
                lane.append(-2)
                continue
            pos = np.mean(pos)
            lane.append(round(pos))
        lanes.append(lane)
    tmp_dict['lanes'] = lanes
    tmp_dict['h_samples'] = h_samples
    tmp_dict['raw_file'] = name
    tmp_dict['run_time'] = 10
    
    json_str = json.dumps(tmp_dict)
    fp.write(json_str+'\n')
fp.close()
print("train_len:{}".format(len(list_train)))
print("val_len:{}".format(len(list_val)))
#train_len:2107
#val_len:522