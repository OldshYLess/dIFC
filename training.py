import os, sys
import numpy as np
import cv2
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from data_train import train_data, High_Data, Low_Data
from model_cycle import High2Low, Discriminator
from model import GEN_DEEP
from dataset import get_loader

import argparse

#parser = argparse.ArgumentParser()
#parser.add_argument("-c", "--gpu", action="store", dest="gpu", help="separate numbers with commas, eg. 3,4,5", required=True)


if __name__ == "__main__":
    # parser for arguments
    #args = parser.parse_args()
    #print(args.gpu)
    #gpu
    #args.gpu=0
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    #gpus = args.gpu.split(",")
    #n_gpu = len(gpus)
    # seed_num?
    seed_num = 2021
    random.seed(seed_num)
    np.random.seed(seed_num)
    torch.manual_seed(seed_num)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    save_epochLoss=True
    save_iterationLoss=True

    max_epoch = 30
    learn_rate = 1e-4
    alpha, beta = 1, 0.05
    b_size=4

    G_h2l = High2Low().cuda()
    D_h2l = Discriminator(32).cuda()
    G_l2h = GEN_DEEP().cuda()
    D_l2h = Discriminator(128).cuda()
    mse = nn.MSELoss()

    optim_D_h2l = optim.Adam(filter(lambda p: p.requires_grad, D_h2l.parameters()), lr=learn_rate, betas=(0.0, 0.9))
    optim_G_h2l = optim.Adam(G_h2l.parameters(), lr=learn_rate, betas=(0.0, 0.9))
    optim_D_l2h = optim.Adam(filter(lambda p: p.requires_grad, D_l2h.parameters()), lr=learn_rate, betas=(0.0, 0.9))
    optim_G_l2h = optim.Adam(G_l2h.parameters(), lr=learn_rate, betas=(0.0, 0.9))

    data = train_data(High_Data, Low_Data)
    loader = DataLoader(dataset=data, batch_size=b_size, shuffle=True)
    # loader dir changed here
    LR_test_loader = get_loader("cells", bs=1)
    HR_test_loader = get_loader("HR_test", bs=1)
    num_test = 200
    test_save = "intermid_results"
    loss_G_h2l_epoch_arr=[]
    loss_G_l2h_epoch_arr=[]
    loss_D_h2l_epoch_arr=[]
    loss_D_l2h_epoch_arr=[]
    for ep in range(1, max_epoch+1):
        G_h2l.train()
        D_h2l.train()
        G_l2h.train()
        D_l2h.train()
        loss_G_h2l_arr=[]
        loss_G_l2h_arr=[]
        loss_D_h2l_arr=[]
        loss_D_l2h_arr=[]
        for i, batch in enumerate(loader):
            optim_D_h2l.zero_grad()
            optim_D_l2h.zero_grad()
            optim_G_h2l.zero_grad()
            optim_G_l2h.zero_grad()

            zs = batch["z"].cuda()
            lrs = batch["lr"].cuda()
            hrs = batch["hr"].cuda()
            downs = batch["hr_down"].cuda()
            #print("------",zs)
            lr_gen = G_h2l(hrs, zs)
            lr_gen_detach = lr_gen.detach()
            hr_gen = G_l2h(lr_gen_detach)
            hr_gen_detach = hr_gen.detach()

            # update discriminator
            loss_D_h2l = nn.ReLU()(1.0 - D_h2l(lrs)).mean() + nn.ReLU()(1 + D_h2l(lr_gen_detach)).mean()
            loss_D_l2h = nn.ReLU()(1.0 - D_l2h(hrs)).mean() + nn.ReLU()(1 + D_l2h(hr_gen_detach)).mean()
            loss_D_h2l.backward()
            loss_D_l2h.backward()
            optim_D_h2l.step()
            optim_D_l2h.step()

            # update generator
            optim_D_h2l.zero_grad()
            gan_loss_h2l = -D_h2l(lr_gen).mean()
            mse_loss_h2l = mse(lr_gen, downs)

            loss_G_h2l = alpha * mse_loss_h2l + beta * gan_loss_h2l
            loss_G_h2l.backward()
            optim_G_h2l.step()

            optim_D_l2h.zero_grad()
            gan_loss_l2h = -D_l2h(hr_gen).mean()
            mse_loss_l2h = mse(hr_gen, hrs)

            loss_G_l2h = alpha * mse_loss_l2h + beta * gan_loss_l2h
            loss_G_l2h.backward()
            optim_G_l2h.step()
            
            loss_G_h2l_arr=np.append(loss_G_h2l_arr,float(loss_G_h2l))
            loss_G_l2h_arr=np.append(loss_G_l2h_arr,float(loss_G_l2h))
            loss_D_h2l_arr=np.append(loss_D_h2l_arr,float(loss_D_h2l))
            loss_D_l2h_arr=np.append(loss_D_l2h_arr,float(loss_D_l2h))
            if save_iterationLoss:
                np.save("iLoss_G_h2l",loss_G_h2l_arr)
                np.save("iLoss_G_l2h",loss_G_l2h_arr)
                np.save("iLoss_D_h2l",loss_D_h2l_arr)
                np.save("iLoss_D_l2h",loss_D_l2h_arr)
            print(" {}({}) D_h2l: {:.3f}, D_l2h: {:.3f}, G_h2l: {:.3f}, G_l2h: {:.3f} \r".format(i+1, ep, loss_D_h2l.item(), loss_D_l2h.item(), loss_G_h2l.item(), loss_G_l2h.item()), end=" ")
            print()
        loss_G_h2l_avg=np.mean(loss_G_h2l_arr)
        loss_G_l2h_avg=np.mean(loss_G_l2h_arr)  
        loss_D_h2l_avg=np.mean(loss_D_h2l_arr)
        loss_D_l2h_avg=np.mean(loss_D_l2h_arr)
      
        print("\n Testing and saving...")
        loss_G_h2l_epoch_arr=np.append(loss_G_h2l_epoch_arr,loss_G_h2l_avg)
        loss_G_l2h_epoch_arr=np.append(loss_G_l2h_epoch_arr,loss_G_l2h_avg)
        loss_D_h2l_epoch_arr=np.append(loss_D_h2l_epoch_arr,loss_D_h2l_avg)
        loss_D_l2h_epoch_arr=np.append(loss_D_l2h_epoch_arr,loss_D_l2h_avg)
        G_h2l.eval()
        D_h2l.eval()
        G_l2h.eval()
        D_l2h.eval()
        for i, sample in enumerate(LR_test_loader):
            if i >= num_test: 
                break
            low_temp = sample["img32"].numpy()
            
            low = torch.from_numpy(np.ascontiguousarray(low_temp[:, ::-1, :, :])).cuda()
            with torch.no_grad():
                hign_gen = G_l2h(low)
            np_low = low.cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
            np_gen = hign_gen.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
            np_low = (np_low - np_low.min()) / (np_low.max() - np_low.min())
            np_gen = (np_gen - np_gen.min()) / (np_gen.max() - np_gen.min())
            np_low = (np_low * 255).astype(np.uint8)
            np_gen = (np_gen * 255).astype(np.uint8)
            cv2.imwrite("{}/HR_Gen_imgs/{}_{}_lr.png".format(test_save, ep, i+1), np_low)
            cv2.imwrite("{}/HR_Gen_imgs/{}_{}_gen.png".format(test_save, ep, i+1), np_gen)
        for i, sample in enumerate(HR_test_loader):
            if i >= num_test: 
                break
            high_temp = sample["img32"].numpy()
            noise = np.random.randn(1, 1, 128).astype(np.float32)
            noise = torch.from_numpy(noise).cuda()
            high = torch.from_numpy(np.ascontiguousarray(high_temp[:, ::-1, :, :])).cuda()
            with torch.no_grad():
                low_gen = G_h2l(high,noise)
            np_high = high.cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
            np_gen = low_gen.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze(0)
            np_high = (np_high - np_high.min()) / (np_high.max() - np_high.min())
            np_gen = (np_gen - np_gen.min()) / (np_gen.max() - np_gen.min())
            np_high = (np_high * 255).astype(np.uint8)
            np_gen = (np_gen * 255).astype(np.uint8)
            cv2.imwrite("{}/LR_Gen_imgs/{}_{}_hr.png".format(test_save, ep, i+1), np_high)
            cv2.imwrite("{}/LR_Gen_imgs/{}_{}_gen.png".format(test_save, ep, i+1), np_gen)
        save_file = "{}/models/model_epoch_{:03d}.pth".format(test_save, ep)
        torch.save({"G_h2l": G_h2l.state_dict(), "D_h2l": D_h2l.state_dict(),
                    "G_l2h": G_l2h.state_dict(), "D_l2h": D_l2h.state_dict()}, save_file)
        print("saved: ", save_file)
        if save_epochLoss:
            np.save("eLoss_G_h2l",loss_G_h2l_epoch_arr)
            np.save("eLoss_G_l2h",loss_G_l2h_epoch_arr)
            np.save("eLoss_D_h2l",loss_D_h2l_epoch_arr)
            np.save("eLoss_D_l2h",loss_D_l2h_epoch_arr)
    print("finished.")
