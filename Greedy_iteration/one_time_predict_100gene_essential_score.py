#build a model predict 100 gene essential score one times
#three encoder_layers and only one decoder layer
#random select 100 genes

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


def genTrainTest(dat,number_gene):
    N,L=dat.shape
    train_number=int(N*0.7)
    valid_number=int(N*0.85)
    traindat=dat[0:train_number,0:number_gene] 
    trainlabel=dat[0:train_number,number_gene:L]
    validat=dat[train_number:valid_number,0:number_gene] 
    validlabel=dat[train_number:valid_number,number_gene:L]

    return traindat,trainlabel,validat,validlabel


#there are three CNN layer in encoder
#no matter how much layers, when there is enough epochs ,the result is always same.
class AtuoEncoder(nn.Module):
    def __init__(self):
        super(AtuoEncoder,self).__init__()


        self.encod1=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=32,kernel_size=20,stride=1,padding=1)
            )

        self.encod2=nn.Sequential(
            nn.Conv1d(32,64,kernel_size=26,stride=1,padding=1)
            )

        self.encod3=nn.Sequential(
            nn.Conv1d(64,128,kernel_size=31,stride=1,padding=1)
            )
 
        self.decod1=nn.Sequential(
            nn.ConvTranspose1d(128,1,kernel_size=6,stride=3,padding=1)
            )
        

    def forward(self,x):
        maxpool=nn.MaxPool1d(12,stride=7)
        activate=nn.Tanh()
        xnor=nn.BatchNorm1d(1,momentum=0.5)
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
        out=maxpool(out)


        #decoder
        out=self.decod1(out)
        out=xnor(out)
        return out



criterion=nn.MSELoss()
LR=0.05
number_gene=13154




data=np.loadtxt("/mnt/Storage/home/tuser/hanya/predicet_gene_essential_autoencoder/Remove_pan_cancer_Expression_pre_essen/express_predict_ess/gene_expression_crispr_essential_13154_remove_pan_normali_ess_exp.txt",delimiter="\t")
traindat,trainlabel,validat,validlabel=genTrainTest(data,number_gene)
express_gene_order=tuple(np.loadtxt("/mnt/Storage/home/tuser/hanya/predicet_gene_essential_autoencoder/Remove_pan_cancer_Expression_pre_essen/express_predict_ess/expression_gene_order_13154.txt",delimiter="\t"))
ess_gene_order=pd.Series(np.loadtxt("/mnt/Storage/home/tuser/hanya/predicet_gene_essential_autoencoder/Remove_pan_cancer_Expression_pre_essen/express_predict_ess/essential_gene_order_13154.txt",delimiter="\t"))

print (traindat.shape)
validat=Variable(torch.FloatTensor(validat))
validlabel=Variable(torch.FloatTensor(validlabel))
traindat=Variable(torch.FloatTensor(traindat))
trainlabel=Variable(torch.FloatTensor(trainlabel))

traindat=traindat.view(387,1,13154)
validat=validat.view(83,1,13154)
validlabel=validlabel.view(83,1,13154)
trainlabel=trainlabel.view(387,1,13154)

#the variable is used to store the result essential gene order.
random_ess_gene_orde=np.random.permutation(range(0,13154,1))
ess_gene_result=[]

f=open("/mnt/Storage/home/tuser/hanya/predicet_gene_essential_autoencoder/Remove_pan_cancer_Expression_pre_essen/express_predict_ess/greedy_iteration_13154_predict_essential/predict_100_gene_essential_index.txt","w")
for i in random_ess_gene_orde:
    f.write(str(i)+"\t")
f.close()


#for i in range(0,13101,100):
for i in range(0,13101,100):
    print ("The "+str(i)+"th model")
    index=random_ess_gene_orde[i:(i+100)]
    temp_trainlabel=trainlabel[:,:,index]
    temp_validlabel=validlabel[:,:,index]
    model=AtuoEncoder()
    opt_Momentum = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.5) 

    for j in range(50):   
        outs=model(traindat)
        loss=criterion(outs,temp_trainlabel)

        opt_Momentum.zero_grad()
        loss.backward()
        opt_Momentum.step()

        testouts=model(validat)
        test_loss=criterion(testouts,temp_validlabel)
        print ("train_loss:"+str(loss)+"  valid_loss:"+str(test_loss))

        if (j+1) % 5 == 0:
            torch.save(model,"/mnt/Storage/home/tuser/hanya/predicet_gene_essential_autoencoder/Remove_pan_cancer_Expression_pre_essen/express_predict_ess/one_time_predict_100gene_essential/predict_100_gene_essential_onetimes"+str(i)+"_epoch"+str(j))
    

#output the predicted value,and calculate the correlationshi