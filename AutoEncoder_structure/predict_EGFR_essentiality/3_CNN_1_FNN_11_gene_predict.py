
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


def genTrainTest(dat,random_gene_index,EGFR_ess_index):
    N,L=dat.shape
    train_number=int(N*0.7) 
    valid_num=int(N*0.85)
    traindat=dat[0:train_number,random_gene_index] 
    trainlabel=dat[0:train_number,EGFR_ess_index]
    validdat=dat[train_number:valid_num,random_gene_index] 
    validlab=dat[train_number:valid_num,EGFR_ess_index]
    return traindat,trainlabel,validdat,validlab


#there are three CNN layer in encoder
#no matter how much layers, when there is enough epochs ,the result is always same.
class AtuoEncoder(nn.Module):
    def __init__(self):
        super(AtuoEncoder,self).__init__()


        self.encod1=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=32,kernel_size=2,stride=1,padding=1,bias=False)
            )


        self.encod2=nn.Sequential(
            nn.Conv1d(32,64,kernel_size=2,stride=1,padding=1,bias=False)
            )

        self.encod3=nn.Sequential(
            nn.Conv1d(64,128,kernel_size=2,stride=1,bias=False)
            )
        self.fc=nn.Linear(128,1)

 

    def forward(self,x):
        maxpool=nn.MaxPool1d(4,stride=2)
        activate=nn.Tanh()
        xnor3=nn.BatchNorm1d(32,momentum=0.5)
        xnor4=nn.BatchNorm1d(64,momentum=0.5)
        xnor5=nn.BatchNorm1d(128,momentum=0.5)

        #encoder part
        out=self.encod1(x)
        out=activate(xnor3(out))
        out=maxpool(out)

        out=self.encod2(out)
        out=activate(xnor4(out))
        out=maxpool(out)

        out=self.encod3(out)
        out=activate(xnor5(out))

        out=out.view(out.size()[0],128)
        out=self.fc(out)
        out=out.view(out.size()[0],1)

        return out

criterion=nn.MSELoss()
model4=AtuoEncoder()
LR=0.05
opt_Momentum = torch.optim.SGD(model4.parameters(), lr=LR, momentum=0.5) 


#define the position of EGFR in crispr data.
#random select 10 gene used to predict essential score.
#random_gene_index=[random.randint(1,13999) for _ in range(10)]
#print (random_gene_index)
#random_gene_index=[6971, 5330, 5827, 7162, 4191,12818, 2815, 10595, 10191, 3994, 2969]
#random_gene_index.append(12818)#add EGFR expression
random_gene_index=np.linspace(12813, 12824,num=11,dtype=int)
EGFR_ess_index=13999+5019

data=np.loadtxt("/mnt/Storage/home/tuser/hanya/CRISPR_shRNA_TPM_intersection_data_prediect_essential/predict_EGFR_essential_score/crispr_predict_essential_Norm_exp_ess_across_cell_line.txt",delimiter="\t")
traindat,trainlabel,validdat,validlab=genTrainTest(data,random_gene_index,EGFR_ess_index)



traindat=Variable(torch.FloatTensor(traindat))
trainlabel=Variable(torch.FloatTensor(trainlabel))
validdat=Variable(torch.FloatTensor(validdat))
validlab=Variable(torch.FloatTensor(validlab))

traindat=traindat.view(387,1,11)
validdat=validdat.view(83,1,11)
trainlabel=trainlabel.view(387,1)
validlab=validlab.view(83,1)
print (traindat.shape)
print (validdat.shape)

rec=open("/mnt/Storage/home/tuser/hanya/CRISPR_shRNA_TPM_intersection_data_prediect_essential/predict_EGFR_essential_score/CNN_FNN_near_10/CNN_FNN_near_10_record.txt","w")

for i in range(500):   
    outs=model4(traindat)
    loss=criterion(outs,trainlabel)
    print (loss)

    opt_Momentum.zero_grad()
    loss.backward()
    opt_Momentum.step()

    if (i+1) % 5 == 0:
        rec.write(str(i)+"\n")
        rec.write("Train_loss"+str(loss)+"\n")
        testouts=model4(validdat)
        test_loss=criterion(testouts,validlab)
        rec.write("Test_loss"+str(test_loss)+"\n")
        torch.save(model4,"/mnt/Storage/home/tuser/hanya/CRISPR_shRNA_TPM_intersection_data_prediect_essential/predict_EGFR_essential_score/CNN_FNN_near_10/CNN_FNN_near_10_epoch"+str(i))

#output the predicted value,and calculate the correlationship
rec.close()

