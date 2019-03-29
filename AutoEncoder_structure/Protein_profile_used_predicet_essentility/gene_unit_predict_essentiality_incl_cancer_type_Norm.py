# In the traindat, each line is a gene protein level across cell line and the one-hot-coding of cancer type.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import os
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


        self.encod=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=1,kernel_size=2,stride=2)
            )

        self.encod1=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=32,kernel_size=5,stride=1,padding=2)
            )

        self.encod2=nn.Sequential(
            nn.Conv1d(32,64,kernel_size=6,stride=1,padding=2)
            )

        self.encod3=nn.Sequential(
            nn.Conv1d(64,128,kernel_size=6,stride=1,padding=2)
            )

        self.encod4=nn.Sequential(
            nn.Conv1d(128,256,kernel_size=6,stride=1,padding=2)
            )
 



        self.decod1=nn.Sequential(
            nn.ConvTranspose1d(256,128,kernel_size=6,stride=1,padding=2)
            )  
        self.decod2=nn.Sequential(
            nn.ConvTranspose1d(128,64,kernel_size=6,stride=1,padding=2)
            )
        self.decod3=nn.Sequential(
            nn.ConvTranspose1d(64,32,kernel_size=6,stride=1,padding=2)
            )
        self.decod4=nn.Sequential(
            nn.ConvTranspose1d(32,1,kernel_size=5,stride=1,padding=2)
            )    
   

    def forward(self,x):
        maxpool=nn.MaxPool1d(4,stride=2,return_indices=True)
        activate=nn.Tanh()
        xnor3=nn.BatchNorm1d(32,momentum=0.5)
        xnor4=nn.BatchNorm1d(64,momentum=0.5)
        xnor5=nn.BatchNorm1d(128,momentum=0.5)
        xnor6=nn.BatchNorm1d(256,momentum=0.5)

        out=self.encod(x)
        #encoder part
        out=self.encod1(out)
        out=activate(xnor3(out))
        out,indices1=maxpool(out)

        out=self.encod2(out)
        out=activate(xnor4(out))
        out,indices2=maxpool(out)

        out=self.encod3(out)
        out=activate(xnor5(out))
        out,indices3=maxpool(out)


        out=self.encod4(out)
        out=activate(xnor6(out))
        out,indices4=maxpool(out)
        
        #decoder
        unmax=nn.MaxUnpool1d(4,stride=2)

        out=unmax(out,indices4)
        out=activate(xnor6(out))
        out=self.decod1(out)
        

        out=unmax(out,indices3)
        out=activate(xnor5(out))
        out=self.decod2(out)
        

        out=unmax(out,indices2)
        out=activate(xnor4(out))
        out=self.decod3(out)
        

        out=unmax(out,indices1)
        out=activate(xnor3(out))
        out=self.decod4(out)
        
        return out

criterion=nn.MSELoss()
model4=AtuoEncoder()
LR=0.05
number_gene=888
opt_Momentum = torch.optim.SGD(model4.parameters(), lr=LR, momentum=0.5) 
criterion=nn.MSELoss()


data=np.loadtxt("/mnt/Storage/home/tuser/hanya/protein_predict_essentiality/protein_pred_essential_gene_unit_across_Norm_cancer_coed.txt",delimiter="\t")
traindat,trainlabel,validat,validlabel=genTrainTest(data,number_gene)

print (traindat.shape)
print (trainlabel.shape)
print (validat.shape)
print (validlabel.shape)

validat=Variable(torch.FloatTensor(validat))
validlabel=Variable(torch.FloatTensor(validlabel))
traindat=Variable(torch.FloatTensor(traindat))
trainlabel=Variable(torch.FloatTensor(trainlabel))

traindat=traindat.view(149,1,888)
trainlabel=trainlabel.view(149,1,444)
validat=validat.view(32,1,888)
validlabel=validlabel.view(32,1,444)


rec=open("/mnt/Storage/home/tuser/hanya/protein_predict_essentiality/gene_unit/gene_unit_norm_cancer_ker4_str2.txt","a")
genecor_train=np.ones(149).reshape(149,1)
genecor_valid=np.ones(32).reshape(32,1)
for i in range(1000):   
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


        if (i+1) % 50 ==0:
            tlabel=pd.DataFrame(trainlabel.view(149,444).data.numpy())
            tpre=pd.DataFrame(outs.view(149,444).data.numpy())
            t_result=pd.Series(tlabel.corrwith(tpre,axis=1)).values.reshape(149,1)
            genecor_train=np.append(genecor_train,t_result,axis=1)

            tlabel=pd.DataFrame(validlabel.view(32,444).data.numpy())
            tpre=pd.DataFrame(testouts.view(32,444).data.numpy())
            t_result=pd.Series(tlabel.corrwith(tpre,axis=1)).values.reshape(32,1)
            genecor_valid=np.append(genecor_valid,t_result,axis=1)


        torch.save(model4,"mnt/Storage/home/tuser/hanya/protein_predict_essentiality/gene_unit/gene_unit_norm_cancer_kerner_2_epoch"+str(i+500))

#output the predicted value,and calculate the correlationship
rec.close()

fs=open("mnt/Storage/home/tuser/hanya/protein_predict_essentiality/gene_unit/genecor_train_cor.txt","w")
splitter = "\t"
for i in range(genecor_train.shape[0]):
    for j in range(genecor_train.shape[1]):
        fs.write(str(genecor_train[i][j])+"\t")
    fs.write("\n")
fs.close()

fs=open("mnt/Storage/home/tuser/hanya/protein_predict_essentiality/gene_unit/genecor_validation_cor.txt","w")
splitter = "\t"
for i in range(genecor_valid.shape[0]):
    for j in range(genecor_valid.shape[1]):
        fs.write(str(genecor_valid[i][j])+"\t")
    fs.write("\n")
fs.close()
