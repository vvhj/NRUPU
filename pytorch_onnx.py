import torch, os, cv2
from model.model import parsingNet,parsingNet_acpr
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import scipy.special
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor
import time
#import tensorrt as trt
import onnx
import onnxruntime
import netron
from tqdm import tqdm
def get_onnx(model, onnx_save_path, example_tensor):
 
    example_tensor = example_tensor.cuda()
 
    _ = torch.onnx.export(model,  # model being run
                                  example_tensor,  # model input (or a tuple for multiple inputs)
                                  onnx_save_path,
                                  verbose=False,  # store the trained parameter weights inside the model file
                                  training=False,
                                  do_constant_folding=True,
                                  input_names=['input'],
                                  output_names=['output'],
                                  opset_version=10), 
                                  #dynamic_axes={"input":{0: "batch_size"}, "output":{0: "batch_size"},})
    net = onnx.load(onnx_save_path)
    onnx.checker.check_model(net)
    onnx.helper.printable_graph(net.graph) 
    print("ok")
    #netron.start(onnx_save_path)

    session = onnxruntime.InferenceSession(onnx_save_path) 
    out_r = session.run(None, {"input": np.random.rand(1, 3,  288, 800).astype('float32')}) 
    print(len(out_r))
    print(out_r[0].shape)

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    dist_print('start testing...')

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    elif cfg.dataset == 'Yqpd':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    modelname = cfg.test_model
    modelpath = "pytorchpath/"+modelname+".pkl"
    onnx_save_path = "onnx/"+modelname+".onnx"
    example_tensor = torch.randn(1, 3, 288, 800, device='cuda')
    net = torch.load(modelpath)
    net.cuda()
    net.eval()

    t_all = None
    x = torch.zeros((1,3,288,800)).cuda() + 1
    for i in range(50): 
        y = net(x)
    for j in tqdm(range(100)):
        t_all_1 = []
        for i in range(100):
            t1 = time.time()
            y = net(x)
            t2 = time.time()
            t_all_1.append(t2 - t1)
        if t_all==None:
            import copy
            t_all = copy.deepcopy(t_all_1)
        else:
            t_all = list(np.array(t_all)+np.array(t_all_1))
    t_all = list(np.array(t_all)/100)

    print('average fps:',1 / np.mean(t_all))

    print('std fps:',np.std(1 / np.array(t_all)))
    