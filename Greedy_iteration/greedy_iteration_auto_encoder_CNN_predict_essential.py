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
import pandas as pd

#divide the data into three partation.
def genTrainTest(dat,number_gene):
    N,L=dat.shape
    train_number=int(N*0.7)
    valid_number=int(N*0.85)
    traindat=dat[0:train_number,0:number_gene] 
    trainlabel=dat[0:train_number,number_gene:L]
    validat=dat[train_number:valid_number,0:number_gene] 
    validlabel=dat[train_number:valid_number,number_gene:L]

    return traindat,trainlabel,validat,validlabel


#there is the define of Auto-encoder_CNN
#six layer, three encoder CNN and three decoder CNN
class AtuoEncoder(nn.Module):
    def __init__(self,ker1,ker2,ker3,**kw):
        super(AtuoEncoder,self).__init__()
        self.ker1=ker1
        self.ker2=ker2
        self.ker3=ker3


        self.encod1=nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=32,kernel_size=self.ker1,stride=1,padding=1)
            )

        self.encod2=nn.Sequential(
            nn.Conv1d(32,64,kernel_size=self.ker2,stride=1,padding=1)
            )

        self.encod3=nn.Sequential(
            nn.Conv1d(64,128,kernel_size=self.ker3,stride=1,padding=1)
            )
 
        self.decod1=nn.Sequential(
            nn.ConvTranspose1d(128,64,kernel_size=self.ker3,stride=1,padding=1)
            )
        self.decod2=nn.Sequential(
            nn.ConvTranspose1d(64,32,kernel_size=self.ker2,stride=1,padding=1)
            )
        self.decod3=nn.Sequential(
            nn.ConvTranspose1d(32,1,kernel_size=self.ker1,stride=1,padding=1)
            )    
   

    def forward(self,x,kernel_size,unmax_str):
        maxpool=nn.MaxPool1d(kernel_size=kernel_size,stride=unmax_str,return_indices=True)
        activate=nn.Tanh()
        xnor3=nn.BatchNorm1d(32,momentum=0.5)
        xnor4=nn.BatchNorm1d(64,momentum=0.5)
        xnor5=nn.BatchNorm1d(128,momentum=0.5)

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

        #decoder
        unmax=nn.MaxUnpool1d(kernel_size=kernel_size,stride=unmax_str)

        out=unmax(out,indices3)
        out=activate(xnor5(out))
        out=self.decod1(out)

        out=unmax(out,indices2)
        out=activate(xnor4(out))
        out=self.decod2(out)
    
        out=unmax(out,indices1)
        out=activate(xnor3(out))
        out=self.decod3(out)
        
        return out

#load the model which have train before
#model4=AtuoEncoder(nn.Module,ker1=6,ker2=7,ker3=6,unmax_str=7)
model=torch.load("/mnt/Storage/home/tuser/hanya/predicet_gene_essential_autoencoder/Remove_pan_cancer_Expression_pre_essen/express_predict_ess/Normalized_exp_ess_six_layer_mse/wconv_ker_12_deco_ker14_str2_loss_without_NA_epoch3719")
criterion=nn.MSELoss()
LR=0.05
number_gene=13154
#opt_Momentum = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.5) 


#load the input(expression) true output(essential)
# the expression gene order and the essential gene order.  
#tuple‘s which have attribute .index() could be used to find index
#the pd.Series which have attribution of sort_values and reset_index
data=np.loadtxt("/mnt/Storage/home/tuser/hanya/predicet_gene_essential_autoencoder/Remove_pan_cancer_Expression_pre_essen/express_predict_ess/gene_expression_crispr_essential_13154_remove_pan_normali_ess_exp.txt",delimiter="\t")
traindat,trainlabel,validat,validlabel=genTrainTest(data,number_gene)
express_gene_order=tuple(np.loadtxt("/mnt/Storage/home/tuser/hanya/predicet_gene_essential_autoencoder/Remove_pan_cancer_Expression_pre_essen/express_predict_ess/expression_gene_order_13154.txt",delimiter="\t"))
ess_gene_order=pd.Series(np.loadtxt("/mnt/Storage/home/tuser/hanya/predicet_gene_essential_autoencoder/Remove_pan_cancer_Expression_pre_essen/express_predict_ess/essential_gene_order_13154.txt",delimiter="\t"))
# store the result
result_gene_id_list=[] #used to save the order of gene result
result_train_gene_value_list=[] #to save the predict essential score for each gene
result_valid_gene_value_list=[]


validat=Variable(torch.FloatTensor(validat))
validlabel=Variable(torch.FloatTensor(validlabel))
traindat=Variable(torch.FloatTensor(traindat))
trainlabel=Variable(torch.FloatTensor(trainlabel))

traindat=traindat.view(387,1,13154)
trainlabel=trainlabel.view(387,1,13154)
validat=validat.view(83,1,13154)
validlabel=validlabel.view(83,1,13154)
outs=model(traindat,12,7)
loss=criterion(outs,trainlabel)
gene_select=1315
#use the model to predict the essential score in train dataset, and according to the correalation in train data to select the top10 percent
testouts=model(validat,12,7)
test_loss=criterion(testouts,validlabel)
tlabel=pd.DataFrame(trainlabel.view(387,13154).data.numpy())#transfor the data form in order to calculate the correlation
tpre=pd.DataFrame(outs.view(387,13154).data.numpy())
#the correaltion result
t_result=pd.Series(tlabel.corrwith(tpre).values.reshape(13154))
t_result.index=ess_gene_order
t_result_so=t_result.sort_values(ascending=False)

gene_cor_order=pd.Series(t_result_so.index)
top10=gene_cor_order[0:gene_select]#top10 is used to store the gene which is not need to buold a new model

result_gene_id_list.extend(top10)
print (len(result_gene_id_list))
ess_gene_order=tuple(ess_gene_order)
# delete the top10 gene from the essential data and essential gene order file
ess_index=[]
for i in top10:
    ess_index.append(ess_gene_order.index(i))

outs=outs.view(387,13154)
testouts=testouts.view(83,13154)
result_train_gene_value_list.extend(outs[0:387,ess_index])
result_valid_gene_value_list.extend(testouts[0:387,ess_index])


another=gene_cor_order[gene_select:]
exp_index=[]
for i in another:
    exp_index.append(express_gene_order.index(i))
ess_anoth_index=[]
for i in another:
    ess_anoth_index.append(ess_gene_order.index(i))

traindat=traindat.view(387,13154)
trainlabel=trainlabel.view(387,13154)
validat=validat.view(83,13154)
validlabel=validlabel.view(83,13154)

traindat=traindat[0:387,exp_index]
trainlabel=trainlabel[0:387,ess_anoth_index]
validat=validat[0:83,exp_index]
validlabel=validlabel[0:83,ess_anoth_index]

#gene_cor_order=gene_cor_order[exp_index]
ess_gene_order=pd.Series(ess_gene_order)
ess_gene_order=ess_gene_order[ess_anoth_index]
#print (ess_gene_order[1:20])
express_gene_order=pd.Series(express_gene_order)
express_gene_order=express_gene_order[exp_index]
#print (express_gene_order[1:20])
express_gene_order=tuple(express_gene_order)

modellist=["model1","model2","model3","model4","model5","model6","model7","model8","model9"]
ker1_arr=[7,8,8,7,6,5,7,7,7]
ker2_arr=[8,9,6,9,6,3,6,3,5]
ker3_arr=[7,8,5,4,4,4,4,7,9]
max_ker=[12,12,12,12,12,12,10,10,10]
max_str=[7,7,6,6,6,6,5,5,5]
gene_select=1315



for num_ite in range(0,9,1):
    print (num_ite)
    N,L=traindat.shape
 #number of gene which is not need to build model 
    traindat=traindat.view(387,1,L)
    trainlabel=trainlabel.view(387,1,L)
    validat=validat.view(83,1,L)
    validlabel=validlabel.view(83,1,L)
    model_iter=AtuoEncoder(ker1_arr[num_ite],ker2_arr[num_ite],ker3_arr[num_ite])
    opt_Momentum = torch.optim.SGD(model_iter.parameters(), lr=LR, momentum=0.5) 
    for i in range(1000):   
        outs=model_iter(traindat,max_ker[num_ite],max_str[num_ite])
        loss=criterion(outs,trainlabel)
        print ("iteration"+str(num_ite)+": train loss value"+str(loss))
        testouts=model_iter(validat,max_ker[num_ite],max_str[num_ite])
        test_loss=criterion(testouts,validlabel)
        print ("iteration"+str(num_ite)+": valid loss value"+str(test_loss))

        opt_Momentum.zero_grad()
        loss.backward()
        opt_Momentum.step()
        if (i+1) % 20 ==0:
            torch.save(model_iter,"/mnt/Storage/home/tuser/hanya/predicet_gene_essential_autoencoder/Remove_pan_cancer_Expression_pre_essen/express_predict_ess/greed_iteration_record/greed_iteration_"+str(num_ite)+"epoch"+str(i))

    tlabel=pd.DataFrame(trainlabel.view(N,L).data.numpy())#transfor the data form in order to calculate the correlation
    tpre=pd.DataFrame(outs.view(N,L).data.numpy())
#the correaltion result
    t_result=pd.Series(tlabel.corrwith(tpre).values.reshape(L))
    t_result.index=ess_gene_order
    t_result_so=t_result.sort_values(ascending=False)

    gene_cor_order=pd.Series(t_result_so.index)

    if num_ite == 8 :
        gene_select=1319

    top10=gene_cor_order[0:gene_select]#top10 is used to store the gene which is not need to buold a new model

    result_gene_id_list.extend(top10)
    print (len(result_gene_id_list))
    ess_gene_order=tuple(ess_gene_order)
# delete the top10 gene from the essential data and essential gene order file
    ess_index=[]
    for i in top10:
        ess_index.append(ess_gene_order.index(i))

    outs=outs.view(387,L)
    testouts=testouts.view(83,L)
    result_train_gene_value_list.extend(outs[0:387,ess_index])
    result_valid_gene_value_list.extend(testouts[0:83,ess_index])


    another=gene_cor_order[gene_select:]
    exp_index=[]
    for i in another:
        exp_index.append(express_gene_order.index(i))
    ess_anoth_index=[]
    for i in another:
        ess_anoth_index.append(ess_gene_order.index(i))


    if num_ite != 8 :
        traindat=traindat.view(387,L)
        trainlabel=trainlabel.view(387,L)
        validat=validat.view(83,L)
        validlabel=validlabel.view(83,L)
        traindat=traindat[0:387,exp_index]
        trainlabel=trainlabel[0:387,ess_anoth_index]
        validat=validat[0:387,exp_index]
        validlabel=validlabel[0:387,ess_anoth_index]

        ess_gene_order=pd.Series(ess_gene_order)
        ess_gene_order=ess_gene_order[ess_anoth_index]
        #print (ess_gene_order[1:20])


        express_gene_order=pd.Series(express_gene_order)
        express_gene_order=express_gene_order[exp_index]
        #print (express_gene_order[1:20])
        express_gene_order=tuple(express_gene_order)
        
        

f=open("/mnt/Storage/home/tuser/hanya/predicet_gene_essential_autoencoder/Remove_pan_cancer_Expression_pre_essen/express_predict_ess/greed_iteration_record/greed_result_gene_order.txt","w")
for i in result_gene_id_list:
    f.write(str(i)+"\t")
f.close()

f=open("/mnt/Storage/home/tuser/hanya/predicet_gene_essential_autoencoder/Remove_pan_cancer_Expression_pre_essen/express_predict_ess/greed_iteration_record/greed_result_train_gene_essential.txt","w")
for i in result_train_gene_value_list:
    f.write(str(i)+"\t")
f.close()

f=open("/mnt/Storage/home/tuser/hanya/predicet_gene_essential_autoencoder/Remove_pan_cancer_Expression_pre_essen/express_predict_ess/greed_iteration_record/greed_result_valid_gene_essential.txt","w")
for i in result_valid_gene_value_list:
    f.write(str(i)+"\t")
f.close()
