# build a auto-encoder structure to predict the gene essential score
# use the DepMap shRNA data as the train data, and the DepMap CRISPR data as the test data. 
# the encode part we use the convolution structure and the decode part is the deconvolution part.
# the input expression profile gene order is according Hierarchical clustering of TCGA gene profile 
# the output gene order is according the Hierarchical clustering of shRNA and CRISPR value.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import os

# divide the data into expression and essential score
def genTrainTest(dat,number_gene):
    N,L=dat.shape
    traindat=dat[0:N,0:number_gene] 
    trainlabel=dat[0:N,number_gene:L]
    return traindat,trainlabel

# there are three CNN layer in encoder
# in the convolution part this is only have one hide layer and one input/output layer
class AtuoEncoder(nn.Module):
    def __init__(self):
        super(AtuoEncoder,self).__init__()
        #the input in 1*13102 ,output is 8*935
        self.encod1=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=8,kernel_size=13,stride=1),
            nn.Tanh(),
            nn.MaxPool1d(14,stride=14,return_indices=True)
            )
        #the input is 8*935, output is 16*66
        self.encod2=nn.Sequential(
            nn.Conv1d(8,16,kernel_size=12,stride=1),
            nn.Tanh(),
            nn.MaxPool1d(14,stride=14,return_indices=True)
            )
        
        self.decod1=nn.Sequential(
            nn.ConvTranspose1d(16,8,kernel_size=12,stride=1),
            nn.Tanh()
            )
        
        self.decod2=nn.Sequential(
            nn.ConvTranspose1d(8,1,kernel_size=13,stride=1),
            nn.Tanh()
            )
        
        
# the input size x is equal to the size of output.
    def forward(self,x):
        #encoder part
        xnor=nn.BatchNorm1d(1,momentum=0.5)
        xnor2=nn.BatchNorm1d(8,momentum=0.5)
        xnorde=nn.BatchNorm1d(8,momentum=0.5)
        dropout=nn.Dropout(0.2)
        
        #x=xnor(x)
        out,indices1=self.encod1(x)
        out=dropout(out)
        
        #out=xnor2(out)
        out,indices2=self.encod2(out)
        
        #decoder
        unmax=nn.MaxUnpool1d(14,stride=14)
        out=unmax(out,indices2)
        out=self.decod1(out)
        out=dropout(out)
        
        # out=xnorde(out)
        out=unmax(out,indices1)
        out=self.decod2(out)
        
            
        #out,ind1,ind2=self.encoder(x)
        #out=self.decoder(out,ind1,ind2)
        return out
        


model=AtuoEncoder()
LR=0.05
number_gene=13102
opt_Momentum = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.5) 
criterion=nn.MSELoss()

#process the train and test data
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

#because the CNN netowrk the input is three dimension
traindat=traindat.view(431,1,13102)
trainlabel=trainlabel.view(431,1,13102)
testdat=testdat.view(289,1,13102)
testlabel=testlabel.view(289,1,13102)

for i in range(400):   
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
