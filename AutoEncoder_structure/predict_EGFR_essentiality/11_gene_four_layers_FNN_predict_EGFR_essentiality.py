
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


def genTrainTest(dat,number_gene,random_gene_index,EGFR_ess_index):
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
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        input_size=11
        h2_size=7
        h3_size=4
        output_size=1
        self.dropout=nn.Dropout(0.2)

        self.fc=nn.Linear(input_size,h2_size)
        self.norm1 = nn.BatchNorm1d(h2_size,momentum=0.5)
       
        self.fc3=nn.Linear(h2_size,h3_size)#output_size也就是等于number_drugs
        self.norm3 = nn.BatchNorm1d(h3_size,momentum=0.5)

        self.fc4=nn.Linear(h3_size,output_size)#output_size也就是等于number_drugs
        self.norm4 = nn.BatchNorm1d(output_size,momentum=0.5)

    def forward(self,x):
        # encoder partition
        activate=nn.Tanh()
        x=activate(self.norm1(self.dropout(self.fc(x))))
        x=activate(self.norm3(self.dropout(self.fc3(x))))
        x=activate(self.norm4(self.dropout(self.fc4(x))))
        return x

criterion=nn.MSELoss()
model4=Net()
LR=0.05
number_gene=13999
opt_Momentum = torch.optim.SGD(model4.parameters(), lr=LR, momentum=0.5) 


#define the position of EGFR in crispr data.
#random select 10 gene used to predict essential score.
#random_gene_index=[random.randint(1,13999) for _ in range(10)]
#print (random_gene_index)
random_gene_index=[6971, 5330, 5827, 7162, 4191, 2815, 10595, 10191, 3994, 2969]
random_gene_index.append(12818)#add EGFR expression

EGFR_ess_index=13999+5019

data=np.loadtxt("/mnt/Storage/home/tuser/hanya/CRISPR_shRNA_TPM_intersection_data_prediect_essential/predict_EGFR_essential_score/crispr_predict_essential_Norm_exp_ess_across_cell_line.txt",delimiter="\t")
traindat,trainlabel,validdat,validlab=genTrainTest(data,number_gene,random_gene_index,EGFR_ess_index)

print (traindat.shape)
print (validdat.shape)

traindat=Variable(torch.FloatTensor(traindat))
trainlabel=Variable(torch.FloatTensor(trainlabel))
validdat=Variable(torch.FloatTensor(validdat))
validlab=Variable(torch.FloatTensor(validlab))
trainlabel=trainlabel.view(387,1)
validlab=validlab.view(83,1)


rec=open("/mnt/Storage/home/tuser/hanya/CRISPR_shRNA_TPM_intersection_data_prediect_essential/predict_EGFR_essential_score/three_layer_FNN_each_cell_line/three_layer_record.txt","w")

for i in range(3000):   
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
        torch.save(model4,"/mnt/Storage/home/tuser/hanya/CRISPR_shRNA_TPM_intersection_data_prediect_essential/predict_EGFR_essential_score/three_layer_FNN_each_cell_line/three_layer_FNN_epoch"+str(i))

#output the predicted value,and calculate the correlationship
rec.close()

