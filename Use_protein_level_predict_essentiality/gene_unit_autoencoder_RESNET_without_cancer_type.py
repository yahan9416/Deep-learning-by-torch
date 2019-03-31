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
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=None):
        super(ResidualBlock, self).__init__()
        kernel_size=5
        stride=1
        padding=2

        self.con1=nn.Sequential(
            nn.Conv1d(in_channels,out_channels,kernel_size,stride,padding,bias=False)
            )

        self.con2=nn.Sequential(
            nn.Conv1d(in_channels,out_channels,kernel_size,stride,padding,bias=False)
            )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        residual = x
        out = self.con1(x)
        out = self.bn2(out)
        out = self.tanh(out)
        out = self.con2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.tanh(out)
        return out



class AtuoEncoder(nn.Module):
    def __init__(self,block):
        super(AtuoEncoder,self).__init__()

        self.encod1=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=32,kernel_size=5,stride=1,padding=2,bias=False)
            )

        self.layer1 = self.make_layer(block,32, 32)

        self.encod2=nn.Sequential(
            nn.Conv1d(32,64,kernel_size=6,stride=1,padding=2,bias=False)
            )
        
        self.layer2 = self.make_layer(block,64, 64)

        self.encod3=nn.Sequential(
            nn.Conv1d(64,128,kernel_size=6,stride=1,padding=2,bias=False)
            )

        self.decod2=nn.Sequential(
            nn.ConvTranspose1d(128,64,kernel_size=6,stride=1,padding=2,bias=False)
            )
        self.decod3=nn.Sequential(
            nn.ConvTranspose1d(64,32,kernel_size=6,stride=1,padding=2,bias=False)
            )
        self.decod4=nn.Sequential(
            nn.ConvTranspose1d(32,1,kernel_size=5,stride=1,padding=2,bias=False)
            ) 
        self.tanh = nn.Tanh()   
   

    def make_layer(self, block,in_channels,out_channels):
        downsample = None
        layers = []
        layers.append(block(in_channels,out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        maxpool=nn.MaxPool1d(4,stride=2,return_indices=True)
        xnor3=nn.BatchNorm1d(32,momentum=0.5)
        xnor4=nn.BatchNorm1d(64,momentum=0.5)
        xnor5=nn.BatchNorm1d(128,momentum=0.5)

        out=self.encod1(x)
        out=self.tanh(xnor3(out))
        out,indices1=maxpool(out)
        out = self.layer1(out)

        out=self.encod2(out)
        out=self.tanh(xnor4(out))
        out,indices2=maxpool(out)
        out = self.layer2(out)

        out=self.encod3(out)
        out=self.tanh(xnor5(out))
        out,indices3=maxpool(out)

        #decoder partition
        unmax=nn.MaxUnpool1d(4,stride=2)

        out=unmax(out,indices3)
        out=self.tanh(xnor5(out))
        out=self.decod2(out)
        out = self.layer2(out)

        out=unmax(out,indices2)
        out=self.tanh(xnor4(out))
        out=self.decod3(out)
        out = self.layer1(out)


        out=unmax(out,indices1)
        out=self.tanh(xnor3(out))
        out=self.decod4(out)

        return out


criterion=nn.MSELoss()
model4=AtuoEncoder(ResidualBlock)
LR=0.05
number_gene=444
opt_Momentum = torch.optim.SGD(model4.parameters(), lr=LR, momentum=0.5) 
criterion=nn.MSELoss()


data=np.loadtxt("/mnt/Storage/home/tuser/hanya/protein_predict_essentiality/protein_pred_essential_gene_unit_across_Norm.txt",delimiter="\t")
traindat,trainlabel,validat,validlabel=genTrainTest(data,number_gene)

print (traindat.shape)
print (trainlabel.shape)
print (validat.shape)
print (validlabel.shape)

validat=Variable(torch.FloatTensor(validat))
validlabel=Variable(torch.FloatTensor(validlabel))
traindat=Variable(torch.FloatTensor(traindat))
trainlabel=Variable(torch.FloatTensor(trainlabel))

traindat=traindat.view(147,1,444)
trainlabel=trainlabel.view(147,1,444)
validat=validat.view(31,1,444)
validlabel=validlabel.view(31,1,444)


rec=open("/mnt/Storage/home/tuser/hanya/protein_predict_essentiality/auto_RESNET_gene_unit_without_cancer_Tanh/gene_unit_norm_cancer_ker4_str2_add.txt","a")
genecor_train=np.ones(147).reshape(147,1)
genecor_valid=np.ones(31).reshape(31,1)
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


        if (i+1) % 10 ==0:
            tlabel=pd.DataFrame(trainlabel.view(147,444).data.numpy())
            tpre=pd.DataFrame(outs.view(147,444).data.numpy())
            t_result=pd.Series(tlabel.corrwith(tpre,axis=1)).values.reshape(147,1)
            genecor_train=np.append(genecor_train,t_result,axis=1)

            tlabel=pd.DataFrame(validlabel.view(31,444).data.numpy())
            tpre=pd.DataFrame(testouts.view(31,444).data.numpy())
            t_result=pd.Series(tlabel.corrwith(tpre,axis=1)).values.reshape(31,1)
            genecor_valid=np.append(genecor_valid,t_result,axis=1)


        torch.save(model4,"/mnt/Storage/home/tuser/hanya/protein_predict_essentiality/auto_RESNET_gene_unit_without_cancer_Tanh/gene_unit_norm_cancer_kerner_2_epoch"+str(i))

#output the predicted value,and calculate the correlationship
rec.close()

fs=open("/mnt/Storage/home/tuser/hanya/protein_predict_essentiality/auto_RESNET_gene_unit_without_cancer_Tanh/genecor_train_cor.txt","w")
splitter = "\t"
for i in range(genecor_train.shape[0]):
    for j in range(genecor_train.shape[1]):
        fs.write(str(genecor_train[i][j])+"\t")
    fs.write("\n")
fs.close()

fs=open("/mnt/Storage/home/tuser/hanya/protein_predict_essentiality/auto_RESNET_gene_unit_without_cancer_Tanh/genecor_validation_cor.txt","w")
splitter = "\t"
for i in range(genecor_valid.shape[0]):
    for j in range(genecor_valid.shape[1]):
        fs.write(str(genecor_valid[i][j])+"\t")
    fs.write("\n")
fs.close()
