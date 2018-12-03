import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np


def genTrainTest(dat):
    N,L=dat.shape #return the size of dat
    print dat.shape
    np.random.seed(10)
    np.random.shuffle(dat) #random the data by row
    traindat=dat[:226,:L-3] #python's index is from 0 to length-1 ,and if we write :336 is from 0 to 335
    trainlabel=dat[:226,L-3:L]#use 80% sample as train data and others is test.
    testdat=dat[226:,:L-3]
    testlabel=dat[226:,L-3:L]
    return traindat,trainlabel,testdat,testlabel

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.fc=nn.Linear(224,160)
        self.fc2=nn.Linear(160,100)
        self.fc3=nn.Linear(100,40)
        self.fc4=nn.Linear(40,3)
        
    def forward(self,x):
        '''
        softmax 相当于把你的一系列输入它的数值根据分布给出它的可能性,
        就比如对第k类中的东西，在经过softmax的转换之后将拥有最大的概率值。
        而log_softmax函数其实就是对softmax做了log转换
        对于不同的dim,官方的解释是：A dimension along which log_softmax will be computed.
        但其实设置为1的时候和默认的dim=None.
        '''
        x=F.relu(self.fc(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.log_softmax(self.fc4(x),dim=1)
        return x
model=Net()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.5)
'''
对于NLLLoss的损失函数而言，得到的值越小，说明我们的分类结果更好。对于NLLLoss(input,target,weight)而言
其中input是(N,C)的其中N是样本数，而C则是number of feature.
其中target是（N），其中值得范围是0<= value <= N-1.
其中weight: a manual rescaling weight given to each class. If given, has to be a Tensor of size C.
所以其实你在降维的最后的维数其实和你的类别的数目没有太大的关系的
在设置权重的时候，我们可以适当的给样本量小的类别给与打的weight.
'''
criterion=nn.NLLLoss()
if __name__=="__main__":
    data=np.loadtxt("/Users/wubingzhang/Desktop/predict_three_immune_subtye/pytroch_predict_immune_phnotype/three_classific_softmax.txt")
    traindat,trainlabel,testdat,testlabel=genTrainTest(data)
    print traindat.shape
    print trainlabel.shape
#对于通常设置的epochs的一般情况下代表进行多少次循环的优化，
#但在最后的结果中我们有的时候会发现并不是epochs越大，我们在分类的时候得到的准确率就越高，
#相反有可能存在着我们在测试集中过分多次的循环，会导致对训练数据的过拟合，而在测试数据中不能得到很好的准确率。
for i in range(100):
    traindat=Variable(torch.FloatTensor(traindat))
    trainlabel=Variable(torch.from_numpy(trainlabel)).type(torch.uint8)
         
    
    outs=model(traindat)
    loss=criterion(outs,trainlabel)
    #backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if batch_idx % 20 == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.data[0]))


    
    
