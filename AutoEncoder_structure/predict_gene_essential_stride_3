#build a auto-encoder structure to 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import os


def genTrainTest(dat,number_gene):
    N,L=dat.shape
    traindat=dat[0:N,0:number_gene] 
    trainlabel=dat[0:N,number_gene:L]
    return traindat,trainlabel  

#there are three CNN layer in encoder
#4 layers encoder and 4 layers for decoder.
#the internal 128*152. 
#That meaning each sample 13102 gene expression value was transform 128*152 ,then use the 128*152 to predict the essential
#IN Maxpool the stride=3
class AtuoEncoder(nn.Module):
    def __init__(self):
        super(AtuoEncoder,self).__init__()
        #output 16*4361
        self.encod1=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=16,kernel_size=11,stride=1),
            nn.Tanh(),
            nn.MaxPool1d(12,stride=3,return_indices=True)
            )
        #output 32*1447
        self.encod2=nn.Sequential(
            nn.Conv1d(16,32,kernel_size=11,stride=1),
            nn.Tanh(),
            nn.MaxPool1d(13,stride=3,return_indices=True)
            )
        #output 64*1447
        self.encod3=nn.Sequential(
            nn.Conv1d(32,64,kernel_size=11,stride=1),
            nn.Tanh(),
            nn.MaxPool1d(12,stride=3,return_indices=True)
            )
        #output 128*152
        self.encod4=nn.Sequential(
            nn.Conv1d(64,128,kernel_size=11,stride=1),
            nn.Tanh(),
            nn.MaxPool1d(13,stride=3,return_indices=True)
            )
        

        self.decod1=nn.Sequential(
            nn.ConvTranspose1d(128,64,kernel_size=11,stride=1),
            nn.Tanh()
            )
        
        self.decod2=nn.Sequential(
            nn.ConvTranspose1d(64,32,kernel_size=11,stride=1),
            nn.Tanh()
            )
        self.decod3=nn.Sequential(
            nn.ConvTranspose1d(32,16,kernel_size=11,stride=1),
            nn.Tanh()
            )
        
        self.decod4=nn.Sequential(
            nn.ConvTranspose1d(16,1,kernel_size=11,stride=1),
            nn.Tanh()
            )    

    def forward(self,x):
        xnor2=nn.BatchNorm1d(16,momentum=0.5)
        xnor3=nn.BatchNorm1d(32,momentum=0.5)
        xnor4=nn.BatchNorm1d(64,momentum=0.5)
        dropout=nn.Dropout(0.3)
        
        #encoder part
        out,indices1=self.encod1(x)
        out=xnor2(dropout(out))
        out,indices2=self.encod2(out)
        out=xnor3(dropout(out))
        out,indices3=self.encod3(out)
        out=xnor4(dropout(out))
        out,indices4=self.encod4(out)
        
        #decoder
        unmax=nn.MaxUnpool1d(13,stride=3)
        unmax1=nn.MaxUnpool1d(12,stride=3)
        unmax2=nn.MaxUnpool1d(13,stride=3)
        unmax3=nn.MaxUnpool1d(12,stride=3)

        out=unmax(out,indices4)
        out=self.decod1(out)
        out=xnor4(dropout(out))
        
        out=unmax1(out,indices3)
        out=self.decod2(out)
        out=xnor3(dropout(out))

        out=unmax2(out,indices2)
        out=self.decod3(out)
        out=xnor2(dropout(out))

        out=unmax3(out,indices1)
        out=self.decod4(out)
        
            
        #out,ind1,ind2=self.encoder(x)
        #out=self.decoder(out,ind1,ind2)
        return out
        


model=AtuoEncoder()
LR=0.05
number_gene=13102
opt_Momentum = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.5) 
criterion=nn.MSELoss()


data=np.loadtxt("/Users/yahan/Desktop/predicted_gene_essential/gene_expressio_crispr_train_essential.txt",delimiter="\t")
testdat,testlabel=genTrainTest(data,number_gene)
data=np.loadtxt("/Users/yahan/Desktop/predicted_gene_essential/gene_expressio_shrna_test_essential.txt",delimiter="\t")
traindat,trainlabel=genTrainTest(data,number_gene)

print traindat.shape
print trainlabel.shape
print testdat.shape
print testlabel.shape

testdat=Variable(torch.FloatTensor(testdat))
testlabel=Variable(torch.FloatTensor(testlabel))
traindat=Variable(torch.FloatTensor(traindat))
trainlabel=Variable(torch.FloatTensor(trainlabel))

traindat=traindat.view(431,1,13102)
trainlabel=trainlabel.view(431,1,13102)
testdat=testdat.view(289,1,13102)
testlabel=testlabel.view(289,1,13102)

for i in range(3000):   
    outs=model(traindat)
    loss=criterion(outs,trainlabel)
    
    opt_Momentum.zero_grad()
    loss.backward()
    opt_Momentum.step()

    if (i+1) % 5 == 0:
        print (i)
        print ("Train_loss",loss)
        testouts=model(testdat)
        test_loss=criterion(testouts,testlabel)
        print ("Test_loss",test_loss)


#output the predicted value,and calculate the correlationship

trainlabel=trainlabel.view(431,13102)
outs=outs.view(431,13102)
testlabel=testlabel.view(289,13102)
testouts=testouts.view(289,13102)

fs=open("/Users/yahan/Desktop/predicted_gene_essential/train_essential_train_actual_value_3.txt","w")
subtensor=trainlabel.data.numpy()
splitter = "\t"
for i in range(subtensor.shape[0]):
    for j in range(subtensor.shape[1]):
        fs.write(str(subtensor[i][j])+"\t")
    fs.write("\n")
fs.close()

fs3=open("/Users/yahan/Desktop/predicted_gene_essential/train_essential_train_predict_value_3.txt","w")
subtensor=outs.data.numpy()
splitter = "\t"
for i in range(subtensor.shape[0]):
    for value in subtensor[i]:
        fs3.write(str(value)+"\t")
    fs3.write("\n")
fs3.close()


fs1=open("/Users/yahan/Desktop/predicted_gene_essential/test_essential_test_actual_value_3.txt","w")
subtensor=testlabel.data.numpy()
splitter = "\t"
for i in range(subtensor.shape[0]):
    for value in subtensor[i]:
        fs1.write(str(value)+"\t")
    fs1.write("\n")
fs1.close()



fs2=open("/Users/yahan/Desktop/predicted_gene_essential/test_essential_test_predict_value_3.txt","w")
subtensor=testouts.data.numpy()
splitter = "\t"
for i in range(subtensor.shape[0]):
    for value in subtensor[i]:
        fs2.write(str(value)+"\t")
    fs2.write("\n")

fs2.close()
