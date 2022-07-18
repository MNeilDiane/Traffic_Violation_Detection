import time
import matplotlib.pyplot as plt
import torch.cuda
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# Construct Dataloder
class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None, target_transform=None):#预处理 None
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()# 删除 string 字符串末尾的指定字符，默认为空白符，包括空格、换行符、回车符、制表符。
            word = line.split()
            imgs.append(word[0], int(word[1]))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform

    def get_item(self, index):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)

#Load Dataset And Preprocess
pipline_train = transforms.Compose([
    #随机旋转图片
    transforms.RandomHorizontalFlip(),
    #resize 227x227
    transforms.Resize((227, 227)),
    #ToTensor
    transforms.ToTensor(),#数据shape W，H，C ——> C，H，W
    # 其将每一个数值归一化到[0,1]，其归一化方法比较简单，直接除以255即可
    #正则化
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

pipline_test = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_data = MyDataset('./data/catVSdog/train.txt', transform=pipline_train)
test_data = MyDataset('./data/catVSdog/test.txt', transform=pipline_test)

trainloader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
testloader = DataLoader(dataset=test_data, batch_size=32, shuffle=False)

classes = ('cat', 'dog')

#Definde Model
class AlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super.__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),#局部响应归一化
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 256, 5, padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            #inplace如果设置成true,那么原来的input 也会发生改变
            nn.Linear(in_features=(256*6*6), out_features=500),
            nn.ReLU(),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=500, out_features=20),
            nn.ReLU(),
            nn.Linear(in_features=20, out_features=num_classes),
        )

    def forward(self, x):
        x = self.x
        x = x.view(-1, 256*6*6)# reduce the dimensions for linear layer input
        return self.classifier(x)

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
model = AlexNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Define train loop
def train_runner(model, device, trainloader, optimizer, epoch):
    model.train()
    total = 0
    correct = 0.0

    for i,data in enumerate(trainloader, 0):
        inputs, labels = data
        #把模型部署到device上
        inputs,labels = inputs.to(device),labels.to(device)
        #初始化梯度
        optimizer.zero_grad()
        #保存训练结果
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, labels)
        #获取最大概率结果的下标
        predict = outputs.argmax(dim=1)
        ##dim=1表示返回每一行的最大值对应的列下标
        total += labels.size(0)
        correct += (predict == labels).sum().item()
        #反向传播
        loss.backward()
        #更新参数
        optimizer.step()
        if i%100==0:
            print("Train Epoch{} \t Loss: {:.6f}, accuracy: {:.6f}%"
                  .format(epoch, loss.item(), 100 * (correct / total)))
            Loss.append(loss.item())
            Accuary.append(correct/total)

    return loss.item(), correct/total


# Define test
def test_runner(model, device, testloader):
    #因为调用eval()将不启用 BatchNormalization 和 Dropout, BatchNormalization和Dropout置为False
    model.eval()
    correct = 0.0
    test_loss = 0.0
    total = 0
    with torch.no_grad():
        for data, label in testloader:
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, label).item()
            predict = output.argmax(dim=1)
            total += label.size(0)
            correct += (predict == label).sum().item()
            print("Average Loss:{:.6f}, accuary:{:.6f}, ".format(test_loss/total, 100*correct/total))


# Now go on operation
epoch = 20
Loss = []
Accuary = []
for epoch in range(1, epoch+1):
    print("start_time", time.strptime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
    loss, acc = train_runner(model, device, trainloader, optimizer, epoch)
    Loss.append(loss)
    Accuary.append(acc)
    test_runner(model, device, testloader)
    print("end_time: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), '\n')

print("Finish Training")
plt.subplot(2, 1, 1)
plt.plot(Loss)
plt.title('Loss')
plt.show()
plt.subplot(2, 1, 2)
plt.plot(Accuary)
plt.title('Accuary')
plt.show()


print(model)
torch.save(model, './models/alexnet-catvsdog.pth')














