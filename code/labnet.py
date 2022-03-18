from utils.tools import *
from network import *
import os
import torch
import torch.optim as optim
import time
import numpy as np
import pandas as pd
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F

def get_config():
    config = {
        "alpha": 0.1,
        "theta": 0.2,
        # "optimizer": {"type": optim.SGD, "optim_params": {"lr": 0.005, "weight_decay": 10 ** -5}},
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -6}},
        "info": "[SADH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 64,
        "net": labnet,
        # "dataset": "cifar10",
        #"dataset": "coco",
        #"dataset": "mirflickr",
        "dataset": "nuswide_21",
        "epoch": 15,
        "test_map": 15,
        "save_path": "save/SADH",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:0"),
        "bit_list": [16],
    }
    config = config_dataset(config)
    return config


class LabLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(LabLoss, self).__init__()
        self.iter = 0
    def forward(self, f, cl, u, y, ind, config):
        cl_loss = 2*torch.sum(torch.pow(y-cl,2))/config["n_class"]
        sim_loss =  similar_loss(u, u, y)
        sim_loss_f =  similar_loss(f, f, y)
        quantization_loss = config["alpha"] * (u - u.sign()).pow(2).mean()
        self.iter+=1
        if self.iter%50 == 0:
            print('likelihood_loss_f', sim_loss_f)
            print('likelihood_loss', sim_loss)
            print('quantization_loss', quantization_loss)
            print('cl_loss',cl_loss) 
        return sim_loss_f +  cl_loss + sim_loss + quantization_loss
    

    
def cos_sim(a, b):
    a_norm = 1/torch.unsqueeze(torch.sqrt(torch.sum(a*a,dim=1)),dim=0).transpose(1,0).expand(a.size())
    b_norm = 1/torch.unsqueeze(torch.sqrt(torch.sum(b*b,dim=1)),dim=0).transpose(1,0).expand(b.size())
    a_norm = torch.where(torch.isnan(a_norm), torch.full_like(a_norm, 0), a_norm)
    b_norm = torch.where(torch.isnan(b_norm), torch.full_like(b_norm, 0), b_norm)
    cos_sim = (a*a_norm) @ (b*b_norm).t()
    return cos_sim

def similar_loss(config,a,b,targets):
    n = a.size(0)
    bits = a.size(1)
    mask = (targets.matmul(targets.t())>0).type(torch.cuda.FloatTensor)
    theta = config['theta']
    inner_product = cos_sim(a,b)
    mask2 = ((theta-inner_product)>0).type(torch.cuda.FloatTensor)
    mask3 = ((theta+inner_product)>0).type(torch.cuda.FloatTensor)
    similar_loss = torch.mean(torch.mul(1-mask,torch.mul(mask3, theta+inner_product)))+ torch.mean(torch.mul(mask,torch.mul(mask2,(theta-inner_product))))
    return similar_loss

def train_val(config, bit):
    device = config["device"]
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    net = labnet(config["n_class"],bit).to(device)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.005, eps=1e-08, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[2, 3, 4, 5, 6], gamma=0.1)

    criterion = LabLoss(config, bit)

    Best_mAP = 0

    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        net.eval()

        train_loss = 0
        for _, label, ind in train_loader:
            label = label.to(device)

            optimizer.zero_grad()
            f, cl, u = net(label.float())

            loss = criterion(f , cl , u, label.float(), ind, config)
            train_loss += loss.item()

            loss.backward(retain_graph=True)
            optimizer.step()
        train_loss = train_loss / len(train_loader)
        scheduler.step()
        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        if (epoch + 1) % config["test_map"] == 0:
            # print("calculating test binary code......")
            torch.save(net, str(bit)+'bit'+config["dataset"]+'labnet.pkl')
            tst_binary, tst_label = compute_result_lab(test_loader, net, device=device)

            # print("calculating dataset binary code.......")\
            trn_binary, trn_label = compute_result_lab(dataset_loader, net, device=device)

            # print("calculating map.......")
            mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                             config["topK"])

            print(bit,' bits',mAP,'map')
            

if __name__ == "__main__":
    config = get_config()
    print(config)
    for bit in config["bit_list"]:
        train_val(config, bit)
