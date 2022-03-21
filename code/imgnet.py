from utils.tools_sadh import *
from network import *
import pandas as pd
import os
import torch
import torch.optim as optim
import time
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')

def get_config(alpha,lamda,eta,beta):
    config = {
        "alpha": alpha,
        "lamda": lamda,
        "eta": eta,
        "beta": beta,
        #"optimizer": {"type": optim.SGD, "optim_params": {"lr": 0.0001, "weight_decay": 10 ** -5}},
        "optimizer": {"type": optim.RMSprop, "optim_params": {"lr": 1e-5, "weight_decay": 10 ** -6}},
        "info": "[SADH]",
        "resize_size": 256,
        "crop_size": 224,
        "batch_size": 48,
        "net":ResNet,
        # "dataset": "cifar10",
        #"dataset": "coco",
        #"dataset": "mirflickr",
        "dataset": "nuswide_21",
        "epoch": 30,
        "test_map": 10,
        "save_path": "save/SADH",
        # "device":torch.device("cpu"),
        "device": torch.device("cuda:0"),
        "bit_list": [16,32,48,64],
    }
    config = config_dataset(config)
    return config


class SADHLoss(torch.nn.Module):
    def __init__(self, config, bit):
        super(SADHLoss, self).__init__()
        self.clloss = nn.BCEWithLogitsLoss()
    def forward(self, U, F, Y, f_t, u_t, f, u, cl, y, ind, config):
        
        s = (y @ y.t() > 0).float()
        
        sym_u_loss = sym_sim_loss(u,U,u_t,y,Y)
        asym_u_loss = asym_sim_loss(u,U,u_t,y,Y)
        sym_f_loss = sym_sim_loss(f,F,f_t,y,Y)
        asym_f_loss = asym_sim_loss(f,F,f_t,y,Y)
        cl_loss = self.clloss(cl, y)
        quantization_loss = (u - u.sign()).pow(2).mean()

        return config["lamda"] *(sym_u_loss + asym_u_loss) + config["alpha"] *(sym_f_loss + asym_f_loss) + config["beta"] * quantization_loss + config["eta"] *cl_loss
    
def cos_sim(a, b):
    a_norm = 1/torch.unsqueeze(torch.sqrt(torch.sum(a*a,dim=1)),dim=0).transpose(1,0).expand(a.size())
    b_norm = 1/torch.unsqueeze(torch.sqrt(torch.sum(b*b,dim=1)),dim=0).transpose(1,0).expand(b.size())
    a_norm = torch.where(torch.isnan(a_norm), torch.full_like(a_norm, 0), a_norm)
    b_norm = torch.where(torch.isnan(b_norm), torch.full_like(b_norm, 0), b_norm)
    cos_sim = (a*a_norm) @ (b*b_norm).t()
    return cos_sim

def asym_sim_loss(u,U,u_t,y,Y):
    n = u.size(0)
    bits = u.size(1)
    mask = (y.matmul(Y.t())>0).float()
    theta = cos_sim(u_t, U) #class_num x batch_size
    mask_theta = (theta>0).float()
    theta = torch.mul(mask_theta , theta)
    inner_product = cos_sim(u, U)
    mask2 = ((theta-inner_product)>0).float()
    mask3 = ((inner_product-theta)>0).float()
    similar_loss = torch.mean(torch.mul(1-mask,torch.mul(mask3, inner_product-theta)))+ torch.mean(torch.mul(mask,torch.mul(mask2,(theta-inner_product))))
    return similar_loss

def sym_sim_loss(u,U,u_t,y,Y):
    n = u.size(0)
    bits = u.size(1)
    mask = (y.matmul(y.t())>0).float()
    theta = cos_sim(u_t, u_t)
    mask_theta = (theta>0).type(torch.cuda.FloatTensor)
    theta = torch.mul(mask_theta , theta)
    inner_product = cos_sim(u, u)
    mask2 = ((theta-inner_product)>0).type(torch.cuda.FloatTensor)
    mask3 = ((inner_product-theta)>0).type(torch.cuda.FloatTensor)
    similar_loss = torch.mean(torch.mul(1-mask,torch.mul(mask3, inner_product-theta)))+ torch.mean(torch.mul(mask,torch.mul(mask2,(theta-inner_product))))
    return similar_loss


def train_val(config, bit):
    device = config["device"]9j
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(config)
    config["num_train"] = num_train
    s_net = config["net"](bit).to(device)  
    t_net = torch.load(str(bit)+'bit'+config["dataset"]+'labnet.pkl')
    optimizer = config["optimizer"]["type"](filter(lambda p: p.requires_grad, s_net.parameters()), **(config["optimizer"]["optim_params"]))
    criterion = SADHLoss(config, bit)

    Best_mAP = 0
    Y = torch.zeros(config["num_train"], config["n_class"]).to(config["device"]).float()
    t_net.eval()
    for _, label, ind in train_loader:
        label = label.to(device).float()
        f_t, _, u_t = t_net(label.float())
        Y[ind, :] = label
    Y = torch.from_numpy(np.unique(Y.cpu().numpy(), axis=0)).to(config["device"])
    F, _, U = t_net(Y)

    for epoch in range(config["epoch"]):

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, training...." % (
            config["info"], epoch + 1, config["epoch"], current_time, bit, config["dataset"]), end="")

        s_net.train()
        train_loss = 0
        for image, label, ind in train_loader:
            image = image.to(device)
            label = label.to(device)
            f, u, cl = s_net(image)
            f_t, _, u_t = t_net(label.float())
            loss = criterion(U, F, Y, f_t, u_t, f, u, cl, label.float(), ind, config)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))
        max_map = 0
        if (epoch + 1) % config["test_map"] == 0:
            s_net.eval()
            with torch.no_grad():
                print("calculating test binary code......")

                tst_binary, tst_label = compute_result(test_loader, s_net, device=device)

                # print("calculating dataset binary code.......")\
                trn_binary, trn_label = compute_result(dataset_loader, s_net, device=device)

                # print("calculating map.......")

                mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                                     config["topK"])
                if mAP > max_map:
                    max_map = mAP    
                
                print(bit,' bits',max_map,'map')

if __name__ == "__main__":
    eta = 0.1
    lamda = 1
    alpha = 1
    beta = 0.01
    config = get_config(alpha,lamda,eta,beta)
    print(config)
    train_val(config, bit)

