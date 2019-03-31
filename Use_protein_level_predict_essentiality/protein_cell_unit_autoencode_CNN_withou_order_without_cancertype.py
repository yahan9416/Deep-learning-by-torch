#want to calculate the correlation direct by python.
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


        self.encod1=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=32,kernel_size=6,stride=1,padding=2)
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
            nn.ConvTranspose1d(32,1,kernel_size=4,stride=2)
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


model4=torch.load("/mnt/Storage/home/tuser/hanya/protein_predict_essentiality/order_gene_protein/cell_unit_cnn/withcancer_with_Norm_kerner_epoch1999")
criterion=nn.MSELoss()
LR=0.03
number_gene=215#214 protein feature, and the first column is the caner type of cell line  enocoded by one-hot-encoding
opt_Momentum = torch.optim.SGD(model4.parameters(), lr=LR, momentum=0.5) 



data=np.loadtxt("/mnt/Storage/home/tuser/hanya/protein_predict_essentiality/order_gene_protein/protein_pred_essential_cell_unit_across_within_Norm_cancer_encod_order_pro_gene.txt",delimiter="\t")
print (data.shape)
traindat,trainlabel,validat,validlabel=genTrainTest(data,number_gene)
"""
print (traindat.shape)
print (trainlabel.shape)
print (validat.shape)
print (validlabel.shape)
"""
validat=Variable(torch.FloatTensor(validat))
validlabel=Variable(torch.FloatTensor(validlabel))
traindat=Variable(torch.FloatTensor(traindat))
trainlabel=Variable(torch.FloatTensor(trainlabel))

traindat=traindat.view(310,1,215)
trainlabel=trainlabel.view(310,1,162)
validat=validat.view(67,1,215)
validlabel=validlabel.view(67,1,162)


rec=open("/mnt/Storage/home/tuser/hanya/protein_predict_essentiality/order_gene_protein/cell_unit_cnn/withcancer_with_Norm_kerner_4.txt","a")
genecor_train=np.ones(310).reshape(310,1)
genecor_valid=np.ones(67).reshape(67,1)
for i in range(1000):   
    outs=model4(traindat)
    loss=criterion(outs,trainlabel)
    print (loss)

    opt_Momentum.zero_grad()
    loss.backward()
    opt_Momentum.step()

    if (i+1) % 5 == 0:
        rec.write(str(i+5000)+"\n")
        rec.write("Train_loss"+str(loss)+"\n")
        testouts=model4(validat)
        test_loss=criterion(testouts,validlabel)
        rec.write("Test_loss"+str(test_loss)+"\n")
        torch.save(model4,"/mnt/Storage/home/tuser/hanya/protein_predict_essentiality/order_gene_protein/cell_unit_cnn/withcancer_with_Norm_kerner_epoch"+str(i+2000))
    
    if (i+1) % 10 ==0:
        tlabel=pd.DataFrame(trainlabel.view(310,162).data.numpy())
        tpre=pd.DataFrame(outs.view(310,162).data.numpy())
        t_result=pd.Series(tlabel.corrwith(tpre,axis=1)).values.reshape(310,1)
        genecor_train=np.append(genecor_train,t_result,axis=1)

        tlabel=pd.DataFrame(validlabel.view(67,162).data.numpy())
        tpre=pd.DataFrame(testouts.view(67,162).data.numpy())
        t_result=pd.Series(tlabel.corrwith(tpre,axis=1)).values.reshape(67,1)
        genecor_valid=np.append(genecor_valid,t_result,axis=1)
#output the predicted value,and calculate the correlationship
rec.close()

fs=open("/mnt/Storage/home/tuser/hanya/protein_predict_essentiality/order_gene_protein/cell_unit_cnn/genecor_train_cor.txt","a")
splitter = "\t"
for i in range(genecor_train.shape[0]):
    for j in range(genecor_train.shape[1]):
        fs.write(str(genecor_train[i][j])+"\t")
    fs.write("\n")
fs.close()

fs=open("/mnt/Storage/home/tuser/hanya/protein_predict_essentiality/order_gene_protein/cell_unit_cnn/genecor_validation_cor.txt","a")
splitter = "\t"
for i in range(genecor_valid.shape[0]):
    for j in range(genecor_valid.shape[1]):
        fs.write(str(genecor_valid[i][j])+"\t")
    fs.write("\n")
fs.close()
