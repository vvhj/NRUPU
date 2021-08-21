# DATA
dataset='Yqpd'
data_root = '/root/file/yqfod'
#dataset='Tusimple'
#data_root = '/root/dataset/Tusimple'
# TRAIN
epoch = 100
batch_size = 32
optimizer = 'Adam'    #['SGD','Adam']
# learning_rate = 0.1
learning_rate = 4e-4
weight_decay = 1e-4
momentum = 0.9

scheduler = 'cos'     #['multi', 'cos']
# steps = [50,75]
gamma  = 0.1
warmup = 'linear'
warmup_iters = 100

# NETWORK
backbone = '18'
griding_num = 100
use_aux = False

# LOSS
sim_loss_w = 1.0
shp_loss_w = 0.0

# EXP
note = ''

log_path = "tusimple_prune1"

# FINETUNE or RESUME MODEL PATH
finetune = None
resume = None

# TEST
test_model = "Yqpd60"
test_work_dir = "test"

num_lanes = 4