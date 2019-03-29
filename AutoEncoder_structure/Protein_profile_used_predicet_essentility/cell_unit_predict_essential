# input is all protein level in one cell line
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import os
import tensorflow as tf


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
            nn.Conv1d(in_channels=1,out_channels=32,kernel_size=5,stride=1,padding=2)
            )


        self.encod2=nn.Sequential(
            nn.Conv1d(32,64,kernel_size=5,stride=1,padding=2)
            )

        self.encod3=nn.Sequential(
            nn.Conv1d(64,128,kernel_size=5,stride=1,padding=2)
            )

        self.encod4=nn.Sequential(
            nn.Conv1d(128,256,kernel_size=6,stride=1,padding=2)
            )
 



        self.decod1=nn.Sequential(
            nn.ConvTranspose1d(256,128,kernel_size=4,stride=2,padding=2)
            )  
        self.decod2=nn.Sequential(
            nn.ConvTranspose1d(128,64,kernel_size=4,stride=2,padding=1)
            )
        self.decod3=nn.Sequential(
            nn.ConvTranspose1d(64,32,kernel_size=4,stride=2,padding=1)
            )
        self.decod4=nn.Sequential(
            nn.ConvTranspose1d(32,1,kernel_size=16,stride=4,padding=2)
            )    
   

    def forward(self,x):
        maxpool=nn.MaxPool1d(4,stride=2)
        activate=nn.Tanh()
        xnor3=nn.BatchNorm1d(32,momentum=0.5)
        xnor4=nn.BatchNorm1d(64,momentum=0.5)
        xnor5=nn.BatchNorm1d(128,momentum=0.5)
        xnor6=nn.BatchNorm1d(256,momentum=0.5)

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


        out=self.encod4(out)
        out=activate(xnor6(out))
        out=maxpool(out)
        
        #decoder
        out=activate(xnor6(out))
        out=self.decod1(out)
        
        out=activate(xnor5(out))
        out=self.decod2(out)
        
        out=activate(xnor4(out))
        out=self.decod3(out)
        
        out=activate(xnor3(out))
        out=self.decod4(out)
        
        return out


model4=AtuoEncoder()
criterion=nn.MSELoss()
LR=0.05
number_gene=214
opt_Momentum = torch.optim.SGD(model4.parameters(), lr=LR, momentum=0.5) 



data=np.loadtxt("/mnt/Storage/home/tuser/hanya/protein_predict_essentiality/protein_pred_essential_cell_unit_across_within_Norm.txt",delimiter="\t")
traindat,trainlabel,validat,validlabel=genTrainTest(data,number_gene)

print (traindat.shape)
print (trainlabel.shape)
print (validat.shape)
print (validlabel.shape)

validat=Variable(torch.FloatTensor(validat))
validlabel=Variable(torch.FloatTensor(validlabel))
traindat=Variable(torch.FloatTensor(traindat))
trainlabel=Variable(torch.FloatTensor(trainlabel))

traindat=traindat.view(147,1,214)
trainlabel=trainlabel.view(147,1,162)
validat=validat.view(31,1,214)
validlabel=validlabel.view(31,1,162)


rec=open("/mnt/Storage/home/tuser/hanya/protein_predict_essentiality/cell_unit/without_cancer/without_cancer_kerner_4.txt","w")

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
        testouts=model4(validat)
        test_loss=criterion(testouts,validlabel)
        rec.write("Test_loss"+str(test_loss)+"\n")
        torch.save(model4,"/mnt/Storage/home/tuser/hanya/protein_predict_essentiality/cell_unit/without_cancer/without_cancer_kerner_epoch"+str(i))

#output the predicted value,and calculate the correlationship
rec.close()
