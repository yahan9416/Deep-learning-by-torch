#build a auto-encoder structure with more little stride = 3,,so we can capture more information in the orignal dataset. 
#only in shrna data set ,and devide the data set into train and test data set and then calculate the correlationship 
#for stride 3,two different way,this is little_layer
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

def criterion(actual,pred):
    lossfun=nn.MSELoss()
    m,l,n=actual.shape
    tactual=torch.t(actual.view(m,n))
    tpred=torch.t(pred.view(m,n))
    tpred=tpred-(torch.mean(tpred,dim=1,keepdim=True))
    tactual=tactual-(torch.mean(tactual,dim=1,keepdim=True))

    loss=lossfun(tactual,tpred)
    return loss


#there are three CNN layer in encoder
#no matter how much layers, when there is enough epochs ,the result is always same.
class AtuoEncoder(nn.Module):
    def __init__(self):
        super(AtuoEncoder,self).__init__()


        self.encod1=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=32,kernel_size=16,stride=1)
            )


        self.encod2=nn.Sequential(
            nn.Conv1d(32,64,kernel_size=5,stride=1)
            )

        self.encod3=nn.Sequential(
            nn.Conv1d(64,128,kernel_size=7,stride=1)
            )

        self.encod4=nn.Sequential(
            nn.Conv1d(128,256,kernel_size=7,stride=1)
            )
 



        self.decod1=nn.Sequential(
            nn.ConvTranspose1d(256,128,kernel_size=7,stride=1)
            )  
        self.decod2=nn.Sequential(
            nn.ConvTranspose1d(128,64,kernel_size=7,stride=1)
            )
        self.decod3=nn.Sequential(
            nn.ConvTranspose1d(64,32,kernel_size=5,stride=1)
            )
        self.decod4=nn.Sequential(
            nn.ConvTranspose1d(32,1,kernel_size=16,stride=1)
            )    
   

    def forward(self,x):
        maxpool=nn.MaxPool1d(12,stride=7,return_indices=True)
        activate=nn.Tanh()
        xnor3=nn.BatchNorm1d(32,momentum=0.5)
        xnor4=nn.BatchNorm1d(64,momentum=0.5)
        xnor5=nn.BatchNorm1d(128,momentum=0.5)
        xnor6=nn.BatchNorm1d(256,momentum=0.5)

        #encoder part
        out=self.encod1(x)
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
        unmax=nn.MaxUnpool1d(12,stride=7)

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


model4=AtuoEncoder()
LR=0.05
number_gene=13999
opt_Momentum = torch.optim.SGD(model4.parameters(), lr=LR, momentum=0.5) 



data=np.loadtxt("/mnt/Storage/home/tuser/hanya/CRISPR_shRNA_TPM_intersection_data_prediect_essential/gene_expression_across_cellline_crispr_essential_within_ccellline_quantile_nor.txt",delimiter="\t")
traindat,trainlabel,validat,validlabel=genTrainTest(data,number_gene)

print (traindat.shape)
print (trainlabel.shape)
print (validat.shape)
print (validlabel.shape)

validat=Variable(torch.FloatTensor(validat))
validlabel=Variable(torch.FloatTensor(validlabel))
traindat=Variable(torch.FloatTensor(traindat))
trainlabel=Variable(torch.FloatTensor(trainlabel))

traindat=traindat.view(387,1,13999)
trainlabel=trainlabel.view(387,1,13999)
validat=validat.view(83,1,13999)
validlabel=validlabel.view(83,1,13999)


rec=open("/mnt/Storage/home/tuser/hanya/CRISPR_shRNA_TPM_intersection_data_prediect_essential/after_express_across_cellline_essential_within_cellline/nor_loss_mse_without_std/nor_crispr_loss_without_std_kerner_12_7.txt","w")

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
        torch.save(model4,"/mnt/Storage/home/tuser/hanya/CRISPR_shRNA_TPM_intersection_data_prediect_essential/after_express_across_cellline_essential_within_cellline/nor_loss_mse_without_std/loss_without_std_kerner_12_stride_7_epoch"+str(i))

#output the predicted value,and calculate the correlationship
rec.close()