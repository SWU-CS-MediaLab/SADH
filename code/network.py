import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
import torch
class AlexNet(nn.Module):
    def __init__(self, hash_bit, pretrained=True):
        super(AlexNet, self).__init__()

        model_alexnet = models.alexnet()
        model_alexnet.load_state_dict(torch.load('/home/featurize/data/alexnet.pth'))
        self.features = model_alexnet.features
        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = model_alexnet.classifier[1].weight
        cl1.bias = model_alexnet.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = model_alexnet.classifier[4].weight
        cl2.bias = model_alexnet.classifier[4].bias

        self.hash_layer = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Linear(4096, hash_bit),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.hash_layer(x)
        return x


resnet_dict = {"ResNet18": models.resnet18, "ResNet34": models.resnet34, "ResNet50": models.resnet50(),
               "ResNet101": models.resnet101, "ResNet152": models.resnet152}


class ResNet(nn.Module):
    def __init__(self, hash_bit, res_model="ResNet50"):
        super(ResNet, self).__init__()
        model_resnet = resnet_dict[res_model]
        model_resnet.load_state_dict(torch.load('/home/featurize/data/resnet50.pth'))
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                                            self.layer1, self.layer2, self.layer3)
        
        self.emb_layer = nn.Linear(model_resnet.fc.in_features, model_resnet.fc.in_features)
        self.emb_layer.weight.data.normal_(0, 0.01)
        self.emb_layer.bias.data.fill_(0.0)    
        
        self.cl_layer = nn.Linear(2048, 21)
        self.cl_layer.weight.data.normal_(0, 0.01)
        self.cl_layer.bias.data.fill_(0.0)
        
        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.feature_layers(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        emb = self.emb_layer(x)
        cl = self.cl_layer(emb)
        y = self.hash_layer(x)
        y =  F.tanh(y)
        return emb,  y, cl
    
class labnet(nn.Module):
 
    def __init__(self, num_classes, bits):
        
        super(labnet, self).__init__()         
        self.Linear1 = nn.Linear(num_classes, 4096)
        self.Linear2 = nn.Linear(4096, 2048)
        self.Linear3 = nn.Linear(2048, num_classes)
        self.Linear4 = nn.Linear(2048, bits)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.normal_(m.bias, 0, 0.01)

    def forward(self, x):

        x = self.Linear1(x)
        x = F.relu(x)
        features = self.Linear2(x)
        #features = F.relu(features)     
        hash = self.Linear4(features)
        hash = F.tanh(hash)
        results = self.Linear3(features)

        return features, results, hash