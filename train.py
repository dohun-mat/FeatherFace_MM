from __future__ import print_function
import os
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import torch.utils.data as data
from data import WiderFaceDetection, detection_collate, preproc, cfg_eresnet
from layers.modules import MultiBoxLoss
from layers.functions.prior_box import PriorBox
import time
import datetime
import math
from models.retinaface import RetinaFace
import numpy as np
from thop import profile
from thop import clever_format

parser = argparse.ArgumentParser(description='Retinaface Training')
parser.add_argument('--training_dataset', default='./data/widerface/train/label.txt', help='Training dataset directory')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')

args = parser.parse_args()

def calculate_flops(net,img_dim):
    net = net.cuda()
    device = next(net.parameters()).device
    dummy_input = torch.randn(1, 3, img_dim, img_dim).to(device)
    
    macs, params = profile(net, inputs=(dummy_input,))
    macs, params = clever_format([macs, params], "%.3f")
    
    print('========================')
    print(f'Computational complexity: {macs}')
    print(f'Number of parameters: {params}')
    print('========================')
    return macs, params

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)
cfg = None
if args.network == "eresnet":
    cfg = cfg_eresnet

rgb_mean = (104, 117, 123) # bgr order
num_classes = 2
img_dim = cfg['image_size']
num_gpu = cfg['ngpu']
batch_size = cfg['batch_size']
max_epoch = cfg['epoch']
gpu_train = cfg['gpu_train']

num_workers = args.num_workers
momentum = args.momentum
weight_decay = args.weight_decay
initial_lr = cfg['lr']
gamma = args.gamma
training_dataset = args.training_dataset
save_folder = args.save_folder

net = RetinaFace(cfg=cfg)
print("Printing net...")
calculate_flops(net, img_dim)

if args.resume_net is not None:
    print('Loading resume network...')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if num_gpu > 1 and gpu_train:
    net = torch.nn.DataParallel(net).cuda()
else:
    net = net.cuda()

cudnn.benchmark = True

if cfg['optim'] == 'adamw':
    optimizer = torch.optim.AdamW(net.parameters(), cfg['lr'])
else:
    optimizer = optim.SGD(net.parameters(), lr=initial_lr, momentum=momentum, weight_decay=weight_decay)

scheduler_data = WiderFaceDetection(training_dataset,preproc(img_dim, rgb_mean))
scheduler_dataloader = data.DataLoader(scheduler_data, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate) 

scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=initial_lr, steps_per_epoch=len(scheduler_dataloader), epochs = max_epoch)

criterion = MultiBoxLoss(num_classes, 0.35, True, 0, True, 7, 0.35, False)

priorbox = PriorBox(cfg, image_size=(img_dim, img_dim))
with torch.no_grad():
    priors = priorbox.forward()
    priors = priors.cuda()

def train():
    
    net.train()
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    dataset = WiderFaceDetection(training_dataset,preproc(img_dim, rgb_mean))

    epoch_size = math.ceil(len(dataset) / batch_size)
    max_iter = max_epoch * epoch_size

    stepvalues = (cfg['decay1'] * epoch_size, cfg['decay2'] * epoch_size)
    step_index = 0

    epoch_loss = 0.0
    best_loss = float('inf')
    
    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
    else:
        start_iter = 0

    print('start model')
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator

            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate))
            if (epoch % 10 == 0 and epoch > 0) or (epoch % 5 == 0 and epoch > cfg['decay1']):
                torch.save(net.state_dict(), save_folder + cfg['name']+ '_epoch_' + str(epoch) + '.pth')
            epoch += 1

        load_t0 = time.time() 
        if iteration in stepvalues:
            step_index += 1

        # load train data
        images, targets = next(batch_iterator)
        images = images.cuda()
        targets = [anno.cuda() for anno in targets]

        out = net(images)
               
        # backprop
        optimizer.zero_grad()
        loss_l, loss_c, loss_landm = criterion(out, priors, targets)
        loss = cfg['loc_weight'] * loss_l + loss_c + loss_landm
        loss.backward()
        optimizer.step()
        scheduler.step()

        # 에폭 평균을 구하기 위해 loss 누적
        epoch_loss = epoch_loss + loss.item()

        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        lr = optimizer.param_groups[0]['lr']
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loc: {:.4f} Cla: {:.4f} Landm: {:.4f} || LR: {:.8f} || Batchtime: {:.4f} s || ETA: {}'
              .format(epoch, max_epoch, (iteration % epoch_size) + 1,
              epoch_size, iteration + 1, max_iter, loss_l.item(), loss_c.item(), loss_landm.item(), lr, batch_time, str(datetime.timedelta(seconds=eta))))
        
        # --- 추가된 최소 Loss 저장 로직 ---
        # 1 에폭이 끝나는 시점에 도달했을 때
        if (iteration + 1) % epoch_size == 0:
            avg_epoch_loss = epoch_loss / epoch_size
            # print(f'*** Epoch {epoch} Average Loss: {avg_epoch_loss:.4f} ***')
            
            # 현재 에폭의 평균 Loss가 역대 최소 Loss보다 낮을 경우
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                # print(f'>>> 새로운 최소 Loss 달성! 모델을 저장합니다: {best_loss:.4f}')
                torch.save(net.state_dict(), save_folder + cfg['name'] + '_best.pth')
            
            # 다음 에폭을 위해 누적 변수 초기화
            epoch_loss = 0.0

    torch.save(net.state_dict(), save_folder + cfg['name'] + '_Final.pth')

if __name__ == '__main__':
    train()
