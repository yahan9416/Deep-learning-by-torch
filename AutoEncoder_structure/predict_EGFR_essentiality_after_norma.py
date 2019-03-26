
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import os
import tensorflow as tf
import pandas as pd

#half of cell line is used to train model and another cell line is used to test model
def genTrainTest(dat,number_gene):
    N,L=dat.shape
    train_number=int(N*0.5) 
    traindat=dat[0:train_number,0:number_gene] 
    trainlabel=dat[0:train_number,number_gene:L]
    return traindat,trainlabel


#there are three CNN layer in encoder
#no matter how much layers, when there is enough epochs ,the result is always same.
class AtuoEncoder(nn.Module):
    def __init__(self):
        super(AtuoEncoder,self).__init__()

        self.encod1=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=16,kernel_size=23,stride=5,padding=10,bias=False),
            nn.BatchNorm1d(16,momentum=0.5),
            nn.Tanh(),
            nn.MaxPool1d(15,stride=5)
            )

        self.encod2=nn.Sequential(
            nn.Conv1d(in_channels=16,out_channels=32,kernel_size=22,stride=5,padding=10,bias=False),
            nn.BatchNorm1d(32,momentum=0.5),
            nn.Tanh(),
            nn.MaxPool1d(17,stride=5)
            )

        self.encod3=nn.Sequential(
            nn.Conv1d(in_channels=32,out_channels=64,kernel_size=22,stride=5,padding=10,bias=False),
            nn.BatchNorm1d(64,momentum=0.5),
            nn.Tanh(),
            nn.MaxPool1d(16,stride=5)
            )

        self.encod4=nn.Sequential(
            nn.Conv1d(in_channels=64,out_channels=128,kernel_size=21,stride=5,padding=10,bias=False),
            nn.BatchNorm1d(128,momentum=0.5),
            nn.Tanh(),
            nn.MaxPool1d(15,stride=5)
            )

        self.decod1=nn.Sequential(
            nn.BatchNorm1d(128,momentum=0.5),
            nn.ConvTranspose1d(128,64,kernel_size=5,stride=3,padding=4,bias=False),
            nn.Tanh()
            )  
        self.decod2=nn.Sequential(
            nn.BatchNorm1d(64,momentum=0.5),
            nn.ConvTranspose1d(64,32,kernel_size=5,stride=3,padding=4,bias=False),
            nn.Tanh()
            )
        self.decod3=nn.Sequential(
            nn.BatchNorm1d(32,momentum=0.5),
            nn.ConvTranspose1d(32,16,kernel_size=5,stride=3,padding=4,bias=False),
            nn.Tanh()
            )
        self.decod4=nn.Sequential(
            nn.BatchNorm1d(16,momentum=0.5),
            nn.ConvTranspose1d(16,1,kernel_size=5,stride=2,padding=1,bias=False),
            nn.Tanh()
            )  

    def forward(self,x):
        #encoder part
        out=self.encod1(x)
        out=self.encod2(out)
        out=self.encod3(out)
        out=self.encod4(out)
        #decoder part
        out=self.decod1(out)
        out=self.decod2(out)
        out=self.decod3(out)
        out=self.decod4(out)
        return out

criterion=nn.MSELoss()
model4=AtuoEncoder()
LR=0.05
number_sample=13999
opt_Momentum = torch.optim.SGD(model4.parameters(), lr=LR, momentum=0.5) 



data=np.loadtxt("/mnt/Storage/home/tuser/hanya/CRISPR_shRNA_TPM_intersection_data_prediect_essential/predict_EGFR_essential_score/crispr_predict_EGFR_ess_Norm.txt",delimiter="\t")
traindat,trainlabel=genTrainTest(data,number_sample)
del data

print (traindat.shape)
print (trainlabel.shape)


traindat=Variable(torch.FloatTensor(traindat))
trainlabel=Variable(torch.FloatTensor(trainlabel))


traindat=traindat.view(1,1,3877723)
trainlabel=trainlabel.view(1,1,277)

for i in range(3000):   
    outs=model4(traindat)
    loss=criterion(outs,trainlabel)

    opt_Momentum.zero_grad()
    loss.backward()
    opt_Momentum.step()

    if (i+1) % 5 == 0:
        print (str(i)+"\n")
        print ("Train_loss"+str(loss)+"\n")
        torch.save(model4,"/mnt/Storage/home/tuser/hanya/CRISPR_shRNA_TPM_intersection_data_prediect_essential/predict_EGFR_essential_score/norm/predict_EGFR_ess_epoch"+str(i))



