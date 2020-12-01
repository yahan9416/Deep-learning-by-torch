import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
'''
下面哟啊开始使用神经网络来降维了，就是通过输入基因的表达值，然后将200多个药物一起预测cell line 对这些药物的反应
'''

def genTrainTest(dat,number_drug):
    N,L=dat.shape #return the size of dat
    print dat.shape
    np.random.seed(10)
    np.random.shuffle(dat) #random the data by row
    number_train=int(N*0.8)
    traindat=dat[:number_train,:(L-number_drug)] #python's index is from 0 to length-1 ,and if we write :336 is from 0 to 335
    trainlabel=dat[:number_train,(L-number_drug):L]#use 80% sample as train data and others is test.
    testdat=dat[number_train:,:(L-number_drug)]
    testlabel=dat[number_train:,(L-number_drug):L]
    return traindat,trainlabel,testdat,testlabel

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        input_size=16847
        h1_size=12000
        h2_size=8000
        h3_size=5000
        h4_size=3000
        h5_size=1500
        h6_size=800
        output=221
        self.fc=nn.Linear(input_size,h1_size)
        self.fc2=nn.Linear(h1_size,h2_size)
        self.fc3=nn.Linear(h2_size,h3_size)
        self.fc4=nn.Linear(h3_size,h4_size)
        self.fc5=nn.Linear(h4_size,h5_size)
        self.fc6=nn.Linear(h5_size,h6_size)
        self.fc7=nn.Linear(h6_size,output)#output_size也就是等于number_drugs
        
    def forward(self,x):
        x=F.relu(self.fc(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.relu(self.fc4(x))
        x=F.relu(self.fc5(x))
        x=F.relu(self.fc6(x))
        x=self.fc7(x)
        return x

number_drug=221
LR=0.05

model=Net()
#其中对于优化器而言，我们可以在试过之后做出选择嘛，还有一个就是优化器的参数设置，其中的lr是learning rate
opt_Momentum = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.5) 
criterion=nn.MSELoss()
'''
MSELoss：Creates a criterion that measures the mean squared error between n elements in the input x and target y.
其中的输入时两个相同维数的tensor
'''
if __name__=="__main__":
    data=np.loadtxt("/Users/yahan/Desktop/predict_drug_response/predicte_drug_response_expression_drug.txt",delimiter="\t")
    print data.shape
    traindat,trainlabel,testdat,testlabel=genTrainTest(data,number_drug)
    print traindat.shape
    print trainlabel.shape
#对于通常设置的epochs的一般情况下代表进行多少次循环的优化，
#但在最后的结果中我们有的时候会发现并不是epochs越大，我们在分类的时候得到的准确率就越高，
#相反有可能存在着我们在测试集中过分多次的循环，会导致对训练数据的过拟合，而在测试数据中不能得到很好的准确率。
for i in range(1200):
    traindat=Variable(torch.FloatTensor(traindat))
    trainlabel=Variable(torch.FloatTensor(trainlabel))
         
    outs=model(traindat)
    loss=criterion(outs,trainlabel)
    opt_Momentum.zero_grad()
    loss.backward()
    opt_Momentum.step()

    if (i+1) % 100 == 0:
        print ("Train_loss",loss)

model.eval()
testdat=Variable(torch.FloatTensor(testdat))
testlabel=Variable(torch.FloatTensor(testlabel))
testouts=model(testdat)
test_loss=criterion(testouts,testlabel)
print ("Test_loss",test_loss)
