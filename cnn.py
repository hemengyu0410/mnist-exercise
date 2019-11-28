import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import torch.nn.functional as F
import numpy as np
from torchvision import models, transforms
import torchsnooper
learning_rate = 0.01
keep_prob_rate = 0.7 #
max_epoch = 2
BATCH_SIZE = 50

DOWNLOAD_MNIST = False
if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # not mnist dir or mnist is empyt dir
    DOWNLOAD_MNIST = True
train_data = torchvision.datasets.MNIST(root='./mnist/', train=True,
                                            transform=transforms.Compose([transforms.Resize(224),
                                                                          transforms.ToTensor()]),
                                            download=DOWNLOAD_MNIST, )
# train_data.data=torch.unsqueeze(train_data.train_data,dim=1).repeat(1,3,1,1)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
with torch.no_grad():

    test_x = Variable(torch.unsqueeze(test_data.test_data,dim  = 1)).type(torch.FloatTensor)[:500]/255.
    # test_x = Variable(test_data.test_data).type(torch.FloatTensor)[:500] / 255.
    test_x=test_x.repeat(1,3,1,1)
    test_y = test_data.test_labels[:500].numpy()



# print(traindata.train_data)
# traindata.train_data=Variable(torch.unsqueeze(traindata.train_data,dim=1),volatile = True).type(torch.FloatTensor)
# train_data=Variable(train_data.repeat(1,3,1,1))
# # train_data=train_data(transform=torchvision.transforms.ToTensor())
# traindata.train_data=train_data


class VGGNet(nn.Module):
    def __init__(self,num_classes=10):
        super(VGGNet,self).__init__()
        net=models.vgg16(pretrained=True)
        net.classifier=nn.Sequential()
        # net.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.features=net

        self.classifier=nn.Sequential(
                                        nn.Linear(512 * 7 * 7, 4096),  #512 * 7 * 7不能改变 ，由VGG16网络决定的，第二个参数为神经元个数可以微调
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(True),
                nn.Dropout(),
                nn.Linear(4096, num_classes)
        )


    def forward(self,x):
        use_gpu = torch.cuda.is_available()
        if use_gpu:
            x=x.cuda()
        x=self.features(x)
        x=x.view(x.size(0),-1)
        x=self.classifier(x)
        return x

def test(vgg):
    global prediction
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        vgg = vgg.cuda()
    y_pre = vgg(test_x).cpu()

    _,pre_index= torch.max(y_pre,1) #dim=1,返回每一行中最大值的那个元素和索引
    pre_index= pre_index.view(-1)
    prediction = pre_index.data.numpy()
    # prediction = pre_index.data
    correct  = np.sum(prediction == test_y)
    print(prediction)
    print(test_y)
    return correct / 500.0


# @torchsnooper.snoop()
def train(vgg):
    use_gpu = torch.cuda.is_available()
    optimizer = torch.optim.Adam(vgg.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(max_epoch):
        for step, (x_, y_) in enumerate(train_loader):
            x, y = Variable(x_), Variable(y_)
            x=x.repeat(1,3,1,1)
            if use_gpu:
                x=x.cuda()
                y=y.cuda()
                vgg=vgg.cuda()
            # output = vgg(x).cpu()

            output = vgg(x)
            _, pre_index = torch.max(output, 1)  # dim=1,返回每一行中最大值的那个元素和索引
            pre_index = pre_index.view(-1)
            loss = loss_func(output, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step != 0 and step % 20 == 0:
                print("=" * 10, step, "=" * 5, "=" * 5, "test accuracy is ", test(vgg), "=" * 10)


if __name__ == '__main__':

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        vgg = VGGNet().cuda()
    train(vgg)