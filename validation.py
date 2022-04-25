import os
os.sys.path.append(os.getcwd())
import torch
import numpy as np
import torch.backends.cudnn as cudnn
from easydict import EasyDict as edict
from data_test import get_loader
from torch.autograd import Variable
import torchvision.utils as vutils
from model import GEN_DEEP
import cv2
import torch.nn as nn
from collections import OrderedDict
import time
num_test=10000
def to_var(data):
    real_cpu = data
    batchsize = real_cpu.size(0)
    input = Variable(real_cpu.cuda())
    return input, batchsize

def main():
    global index,test_loader,data_low,data_high,img_name
    torch.manual_seed(1)
    np.random.seed(0)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    opt = edict()
    opt.nGPU = 1
    opt.batchsize = 1
    opt.cuda = True
    cudnn.benchmark = True
    print('========================LOAD DATA============================')
    data_name = 'cells'
    
    test_loader = get_loader(data_name, opt.batchsize)
    net_G_low2high = GEN_DEEP().cuda()
    #net_G_low2high = net_G_low2high.cuda()
    #net_dict = torch.load('intermid_results/savemodel/20210618/model_epoch_050.pth')['G_l2h']
    net_dict = torch.load('intermid_results/savemodel/20210326/models/model_epoch_042.pth')['G_l2h']
    #net_dict = torch.load('intermid_results/savemodel/20210404/models/model_epoch_042.pth')['G_l2h']
    # load params
    net_G_low2high.load_state_dict(net_dict)
    net_G_low2high = net_G_low2high.eval()
    index = 0
    test_file = 'test_res_cell/testset'
    
    if not os.path.exists(test_file):
        os.makedirs(test_file)
        
    timepast=[]
    for i, data_dict in enumerate(test_loader):
            start=time.time()            
            index=index+1
            #if i >= num_test: 
                #break
            low_temp = data_dict["img32"].numpy()
            img_name = data_dict['imgpath'][0].split('/')[-1]
            low = torch.from_numpy(np.ascontiguousarray(low_temp[:, ::-1, :, :])).cuda()
            with torch.no_grad():
                hign_gen = net_G_low2high(low)
            np_low = low.cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
            np_gen = hign_gen.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
            np_low = (np_low - np_low.min()) / (np_low.max() - np_low.min())
            np_gen = (np_gen - np_gen.min()) / (np_gen.max() - np_gen.min())
            np_low = (np_low * 255).astype(np.uint8)
            np_gen = (np_gen * 255).astype(np.uint8)
            path = os.path.join(test_file, img_name.split('.')[0]+'.png')
            #print(path)
            cv2.imwrite(path, np_gen)
            end = time.time()
            timepast=np.append(timepast, end-start)
            
            if index % 100==0:
                print("Processing time:", np.mean(timepast))
                print("Writing images:",index)
                timepast=[]
            #cv2.imwrite("{}/imgs/{}_{}_lr.png".format(test_save, ep, i+1), np_low)
            #cv2.imwrite("{}/imgs/{}_{}_sr.png".format(test_save, ep, i+1), np_gen)
if __name__ == '__main__':
    main()
    print("--------Finish!----------")
