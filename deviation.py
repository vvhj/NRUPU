import torch, os, cv2
from utils.common import merge_config
from utils.dist_utils import dist_print
import torch
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms
from data.dataset import LaneTestDataset
from data.constant import culane_row_anchor, tusimple_row_anchor,yq_row_anchor
import time
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    dist_print('start testing...')
    assert cfg.backbone in ['18','18rer','34','50','101','152','50next','101next','50wide','101wide']

    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56
    elif cfg.dataset == 'Yqpd':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError
    import linecache
    
    
    modelname = cfg.test_model#
    modelpath = "pytorchpath/"+modelname+".pkl"
    #torch.save(net,"deviation.pth")
    #net = net.cpu()
    net = torch.load(modelpath)
    net.cuda()
    net.eval()
    from thop import profile
    input_test = torch.randn(1, 3, 288, 800).cuda()
    flops, params = profile(net, inputs=(input_test,))
    print('Params: %.2f' % (params))
    print('Flops: %.2f' % (flops))

    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if cfg.dataset == 'CULane':
        splits = ['test0_normal.txt', 'test1_crowd.txt', 'test2_hlight.txt', 'test3_shadow.txt', 'test4_noline.txt', 'test5_arrow.txt', 'test6_curve.txt', 'test7_cross.txt', 'test8_night.txt']
        datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, 'list/test_split/'+split),img_transform = img_transforms) for split in splits]
        img_w, img_h = 1640, 590
        row_anchor = culane_row_anchor
    elif cfg.dataset == 'Tusimple':
        splits = ['test.txt']
        datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, split),img_transform = img_transforms) for split in splits]
        img_w, img_h = 1280, 720
        row_anchor = tusimple_row_anchor
    elif cfg.dataset == 'Yqpd':
        splits = ['train_gt_val.txt']
        datasets = [LaneTestDataset(cfg.data_root,os.path.join(cfg.data_root, split),img_transform = img_transforms) for split in splits]
        img_w, img_h = 1280, 720
        row_anchor = yq_row_anchor
    else:
        raise NotImplementedError
    timec = []
    cnt = 0
    for split, dataset in zip(splits, datasets):
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        print(split[:-4]+'2.avi')
        vout = cv2.VideoWriter(split[:-4]+'2.avi', fourcc , 10.0, (img_w, img_h))
        linedata = []
        
        for i, data in enumerate(tqdm.tqdm(loader)):
            stime = time.time()
            imgs, names = data
            imgs = imgs.cuda()
            with torch.no_grad():
                out = net(imgs)
            etime = time.time()
            col_sample = np.linspace(0, 800 - 1, cfg.griding_num)
            col_sample_w = col_sample[1] - col_sample[0]


            out_j = out[0].data.cpu().numpy()
            out_j = out_j[:, ::-1, :]
            prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
            idx = np.arange(cfg.griding_num) + 1
            idx = idx.reshape(-1, 1, 1)
            loc = np.sum(prob * idx, axis=0)
            out_j = np.argmax(out_j, axis=0)
            loc[out_j == cfg.griding_num] = 0
            out_j = loc

            line_list = {}
            line_list[0] = []
            line_list[1] = []
            line_list[2] = []
            line_list[3] = []

            # import pdb; pdb.set_trace()
            vis = cv2.imread(os.path.join(cfg.data_root,names[0]))
            for i in range(out_j.shape[1]):
                if i == 0:
                    clor = (255,0,0)
                elif i == 1:
                    clor = (0,255,0)
                elif i == 2:
                    clor = (0,255,255)
                else:
                    clor = (0,0,255)
                if np.sum(out_j[:, i] != 0) > 2:
                    for k in range(out_j.shape[0]):
                        if out_j[k, i] > 0:
                            ppp = (int(out_j[k, i] * col_sample_w * img_w / 800) - 1, int(img_h * (row_anchor[cls_num_per_lane-1-k]/288)) - 1 )
                            cv2.circle(vis,ppp,5,clor,-1)
                            line_list[i].append(ppp)
                        else:
                            line_list[i].append((-2,-2))
                else:
                    for k in range(out_j.shape[0]):
                        line_list[i].append((-2,-2))

            linedata.append(line_list)
            line_list['base_line'] = []
            line_list['disc_line'] = []
            l0 = []
            l1 = []
            l2 = []
            l3 = []
            
            for k in range(out_j.shape[0]):
                if (line_list[0][k]!=(-2,-2)) and (line_list[3][k]!=(-2,-2)) and (line_list[1][k]!=(-2,-2)) and (line_list[2][k]!=(-2,-2)):
                    line_list['base_line'].append((np.array(line_list[0][k])+np.array(line_list[3][k]))/2)
                    line_list['disc_line'].append((np.array(line_list[1][k])+np.array(line_list[2][k]))/2)
                    pp1 = (int(list((np.array(line_list[0][k])+np.array(line_list[3][k]))/2)[0]),int(list((np.array(line_list[0][k])+np.array(line_list[3][k]))/2)[1]))
                    pp2 = (int(list((np.array(line_list[1][k])+np.array(line_list[2][k]))/2)[0]),int(list((np.array(line_list[1][k])+np.array(line_list[2][k]))/2)[1]))
                    #cv2.circle(vis,pp1,5,(128,128,128),-1)
                    #cv2.circle(vis,pp2,5,(12,128,128),-1)
                    l0.append(np.array(line_list[3][k]))
                    l1.append(np.array(line_list[1][k]))
                    l2.append(np.array(line_list[2][k]))
                    l3.append(np.array(line_list[0][k]))
            if(len(l0)==0 or len(l1)==0 or len(l2)==0 or len(l3)==0 ):
                vout.write(vis)
                continue
            bl = np.array(line_list['base_line'])
            dl = np.array(line_list['disc_line'])
            l0 = np.array(l0)
            l3 = np.array(l3)
            l1 = np.array(l1)
            l2 = np.array(l2)
            # x = dl[:,0]-bl[:,0]
            # x0_3 = l3[:,0]-l0[:,0]
            # x1_2 = l2[:,0]-l1[:,0]
            # x_b = np.abs(x0_3) - np.abs(x1_2)
            # valueb = np.sum(x_b)/x_b.shape[0]/2
            # value = np.sum(x)/x.shape[0]/valueb*2
            x0_1 = l1[:,0]-l0[:,0]
            x2_3 = l3[:,0]-l2[:,0]
            v0_1 = np.abs(np.sum(x0_1)/x0_1.shape[0])
            v2_3 = np.abs(np.sum(x2_3)/x2_3.shape[0])
            valuel = v0_1/(v0_1+v2_3)*2
            valuer = v2_3/(v0_1+v2_3)*2

            text = "normal"
            text1 = "normal"
            font = cv2.FONT_HERSHEY_SIMPLEX
            mvr = 0.95
            mvl = 0.95
            if valuer < mvr:
                print("右偏")
                text = "right"+str(valuer)
                #text1 = "right"+str(valueb)
            if valuel < mvl:
                print("左偏")
                text = "left"+str(valuel)
                #text1 = "right"+str(valueb)
            #cv2.putText(vis,text, (200, 200), font, 6, (0,0,255), 1)
            
            #cv2.putText(vis,text1, (200, 400), font, 6, (0,0,255), 1)
            vout.write(vis)
            if cnt%10==0:
                cv2.imwrite("images/{}.png".format(cnt),vis)
            cnt+=1
            timec.append(etime-stime)
        vout.release()
        timec = np.array(timec)
        print(np.mean(timec))