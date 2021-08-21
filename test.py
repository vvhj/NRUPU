from pickle import TRUE
import torch, os, datetime
import numpy as np

from model.model import parsingNet
from data.dataloader import get_train_loader

from utils.dist_utils import dist_print, dist_tqdm, is_main_process, DistSummaryWriter
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import MultiLabelAcc, AccTopk, Metric_mIoU, update_metrics, reset_metrics

from utils.common import merge_config, save_model, save_model_best,cp_projects,save_new_model
from utils.common import get_work_dir, get_logger
from evaluation.eval_wrapper import eval_lane
from tqdm import tqdm
import time

def inference(net, data_label, use_aux):
    if use_aux:
        img, cls_label, seg_label = data_label
        img, cls_label, seg_label = img.cuda(), cls_label.long().cuda(), seg_label.long().cuda()
        cls_out, seg_out = net(img)
        return {'cls_out': cls_out, 'cls_label': cls_label, 'seg_out':seg_out, 'seg_label': seg_label}
    else:
        img, cls_label = data_label
        img, cls_label = img.cuda(), cls_label.long().cuda()
        cls_out = net(img)
        return {'cls_out': cls_out, 'cls_label': cls_label}


def resolve_val_data(results, use_aux):
    results['cls_out'] = torch.argmax(results['cls_out'], dim=1)
    if use_aux:
        results['seg_out'] = torch.argmax(results['seg_out'], dim=1)
    return results


def calc_loss(loss_dict, results, logger, global_step):
    loss = 0

    for i in range(len(loss_dict['name'])):

        data_src = loss_dict['data_src'][i]

        datas = [results[src] for src in data_src]

        loss_cur = loss_dict['op'][i](*datas)

        if global_step % 20 == 0:
            logger.add_scalar('loss/'+loss_dict['name'][i], loss_cur, global_step)

        loss += loss_cur * loss_dict['weight'][i]
    return loss


def train(net, data_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict, use_aux):
    net.train()
    progress_bar = dist_tqdm(data_loader)
    t_data_0 = time.time()
    res = []
    restemp = []
    for b_idx, data_label in enumerate(progress_bar):
        restemp = []
        t_data_1 = time.time()
        reset_metrics(metric_dict)
        global_step = epoch * len(data_loader) + b_idx

        t_net_0 = time.time()
        results = inference(net, data_label, use_aux)

        loss = calc_loss(loss_dict, results, logger, global_step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(global_step)
        t_net_1 = time.time()

        results = resolve_val_data(results, use_aux)
        
        update_metrics(metric_dict, results)
        if global_step % 20 == 0:
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                a = me_op.get()
                logger.add_scalar('metric/' + me_name, me_op.get(), global_step=global_step)
                restemp.append(a)
            res = restemp
        logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        if hasattr(progress_bar,'set_postfix'):
            kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
            progress_bar.set_postfix(loss = '%.3f' % float(loss), 
                                    data_time = '%.3f' % float(t_data_1 - t_data_0), 
                                    net_time = '%.3f' % float(t_net_1 - t_net_0), 
                                    **kwargs)
        t_data_0 = time.time()
        #res.append(restemp)
        
    #res = metric_dict
    return res       


def eval_merge(net, data_loader, loss_dict,logger, epoch, metric_dict, use_aux):
    net.eval()
    progress_bar = dist_tqdm(data_loader)
    t_data_0 = time.time()
    #res = []
    restemp = []
    res = 0
    ll = 0
    for b_idx, data_label in enumerate(progress_bar):
        ll+=1
        restemp = []
        t_data_1 = time.time()
        reset_metrics(metric_dict)
        global_step = epoch * len(data_loader) + b_idx

        t_net_0 = time.time()
        results = inference(net, data_label, use_aux)

        loss = calc_loss(loss_dict, results, logger, global_step)
        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()
        #scheduler.step(global_step)
        t_net_1 = time.time()

        results = resolve_val_data(results, use_aux)
        
        update_metrics(metric_dict, results)
        #if global_step % 20 == 0:
        for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
            a = me_op.get()
            #logger.add_scalar('metric/' + me_name, me_op.get(), global_step=global_step)
            restemp.append(a)
        res += restemp[0]
        #logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        if hasattr(progress_bar,'set_postfix'):
            kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
            progress_bar.set_postfix(loss = '%.3f' % float(loss), 
                                    data_time = '%.3f' % float(t_data_1 - t_data_0), 
                                    net_time = '%.3f' % float(t_net_1 - t_net_0), 
                                    **kwargs)
        t_data_0 = time.time()
        #res.append(restemp)
    res = res/ll
    #res = metric_dict
    return res


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    work_dir = get_work_dir(cfg)

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    dist_print(datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' start training...')
    dist_print(cfg)
    assert cfg.backbone in ['18','18rer','18rerv2','34','50','101','152','50next','101next','50wide','101wide']

    train_loader, cls_num_per_lane = get_train_loader(cfg.batch_size, cfg.data_root, cfg.griding_num, cfg.dataset, cfg.use_aux, distributed, cfg.num_lanes)
    test_loader, cls_num_per_lane = get_train_loader(8, cfg.data_root, cfg.griding_num, cfg.dataset, cfg.use_aux, distributed, cfg.num_lanes)
    modelname = cfg.test_model
    modelpath = "pytorchpath/"+modelname+".pkl"
    #torch.save(net,"deviation.pth")
    #net = net.cpu()
    net = torch.load(modelpath)
    net.cuda()
    net.eval()
    res = eval_lane(net, cfg.dataset, cfg.data_root, cfg.test_work_dir, cfg.griding_num, False, distributed,istrain=False)
    print(res)
    from thop import profile
    input_test = torch.randn(1, 3,288,800).cuda()
    flops, params = profile(net, inputs=(input_test,))
    print('Params: %.2f' % (params))
    print('Flops: %.2f' % (flops))
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
    
    
    