
## CNN model

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/pytorch_cv/images/cnn1.png">

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/pytorch_cv/images/cnn2.png">

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/pytorch_cv/images/cnn3.png">

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/pytorch_cv/images/cnn4.png">

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/pytorch_cv/images/cnn5.png">

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/pytorch_cv/images/cnn6.png">

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/pytorch_cv/images/cnn7.png">

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/pytorch_cv/images/cnn8.png">

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/pytorch_cv/images/cnn9.png">

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/pytorch_cv/images/cnn10.png">

<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/pytorch_cv/images/cnn11.png">


```python
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
import matplotlib.pyplot as plt
```


```python
train_dataset=dsets.MNIST(root="./data",
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)
```


```python
test_dataset=dsets.MNIST(root="./data",
                          train=False,
                          transform=transforms.ToTensor()) # important: ToTensor() not ToTensor, with the ()
```


```python
test_dataset[0][0].size()
```




   torch.Size([1, 28, 28])




```python
show_img=test_dataset[0][0].numpy().reshape(28,28)
```


```python
plt.imshow(show_img, cmap="gray")
plt.show()
```



```python
# labels
test_dataset[0][1]
```




   7



## make dataset iterable
- total: 60000
- batch_size: 100
- iterations: 3000
    - 1 iter: one minibatch forward and backward pass
- epochs
    - 1 epoch: running through the whole dataset once
    - epochs = $$ iter \div \frac{total data}{batchsize}= \frac{iter}{batches}=3000 \div \frac{60000}{100}=5 $$


```python
iter=3000
batch_size=100
num_epochs=iter/(len(train_dataset)/batch_size)
num_epochs
```




   5.0



### create iterable object from training dataset


```python
train_gen=torch.utils.data.DataLoader(dataset=train_dataset,
                                     batch_size=batch_size,
                                     shuffle=True)
```

### check iterability


```python
import collections
isinstance(train_gen, collections.Iterable)
```




   True



### do the same for test dataset (shuffle = False for test data !!)


```python
test_gen=torch.utils.data.DataLoader(dataset=test_dataset,
                                     batch_size=batch_size,
                                     shuffle=False)
```

## build the model


```python
class CNNModel(nn.Module):
    def __init__(self):
        # inheritance
        super(CNNModel,self).__init__()
        # conv 1
        self.conv1=nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1=nn.ReLU()
        # max pool 1
        self.maxpool1=nn.MaxPool2d(kernel_size=2)
        # conv 2
        self.conv2=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1,padding=2)
        self.relu2=nn.ReLU()
        # max pool 2
        self.maxpool2=nn.MaxPool2d(kernel_size=2)
        
        # fc1
        
        self.fc1=nn.Linear(32*7*7, 10)
        
    def forward(self, inputs):
        # conv 1
        net=self.conv1(inputs)
        net=self.relu1(net)
        # max pool1
        net=self.maxpool1(net)
        # conv 2
        net=self.conv2(net)
        net=self.relu2(net)
        # max pool2
        out=self.maxpool2(net)
        
        # resize
        # original size: (100, 32, 7, 7)
        # new size: (100, 32*7*7)
        out=out.view(out.size(0), -1)
        out=self.fc1(out)
        return out
```


```python
model=CNNModel()
```


```python
criterion=nn.CrossEntropyLoss()
```


```python
learning_rate=0.01
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)
```


```python
print(list(model.parameters())[0].size()) # output channels, filters, kernel height, kernel width
print(list(model.parameters())[1].size())
print(list(model.parameters())[2].size())
print(list(model.parameters())[3].size())
print(list(model.parameters())[4].size())
print(list(model.parameters())[5].size())
```

    torch.Size([16, 1, 5, 5])
    torch.Size([16])
    torch.Size([32, 16, 5, 5])
    torch.Size([32])
    torch.Size([10, 1568])
    torch.Size([10])



```python
iter=0
for epoch in range(int(num_epochs)): # epoches=5
    for images, labels in train_gen: # len(train_gen=600)
        # convert inputs to Variables
        if iter==1:
            print(images.size()) # (100, 1, 28, 28)
        images=Variable(images)
        labels=Variable(labels)
         
        # clear grad buffers
        optimizer.zero_grad()
        
        # get outputs with inputs
        outputs=model(images)
        
        # loss
        loss=criterion(outputs, labels)
        
        # backward gradients
        loss.backward()
        
        # updating parameters
        optimizer.step()
        
        # 1 iter: forward + backward pass
        iter+=1
        
        if iter % 500==0:
            correct=0
            total=0
            for images, labels in test_gen:
                images=Variable(images)
                outputs=model(images)
                # images.data.size(): (100, 784)
                # outputs.data.size(): (100, 10)
                _, predicted=torch.max(outputs.data, 1)
                
                total+=labels.size(0)
                
                correct+=(predicted==labels).sum()
            accuracy=correct/total*100
            print("Iteration: {}, Loss: {}, Accurracy: {}".format(iter, loss.data[0], accuracy))
```

    torch.Size([100, 1, 28, 28])
    Iteration: 500, Loss: 0.03004816547036171, Accurracy: 97.92
    Iteration: 1000, Loss: 0.11131425201892853, Accurracy: 97.92999999999999
    Iteration: 1500, Loss: 0.09197033196687698, Accurracy: 97.81
    Iteration: 2000, Loss: 0.07160643488168716, Accurracy: 98.06
    Iteration: 2500, Loss: 0.05089489370584488, Accurracy: 98.24000000000001
    Iteration: 3000, Loss: 0.07023483514785767, Accurracy: 98.25


## more cnn models in pytorch


```python
class CNNModel2(nn.Module):
    def __init__(self):
        # inheritance
        super(CNNModel2,self).__init__()
        # conv 1
        self.conv1=nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1=nn.ReLU()
        # max pool 1
        self.avgpool1=nn.AvgPool2d(kernel_size=2)
        # conv 2
        self.conv2=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1,padding=2)
        self.relu2=nn.ReLU()
        # max pool 2
        self.avgpool2=nn.AvgPool2d(kernel_size=2)
        
        # fc1
        
        self.fc1=nn.Linear(32*7*7, 10)
        
    def forward(self, inputs):
        # conv 1
        net=self.conv1(inputs)
        net=self.relu1(net)
        # max pool1
        net=self.avgpool1(net)
        # conv 2
        net=self.conv2(net)
        net=self.relu2(net)
        # max pool2
        out=self.avgpool2(net)
        
        # resize
        # original size: (100, 32, 7, 7)
        # new size: (100, 32*7*7)
        out=out.view(out.size(0), -1)
        out=self.fc1(out)
        return out
```


```python
model2=CNNModel2()
```


```python
criterion=nn.CrossEntropyLoss()
```


```python
learning_rate=0.01
optimizer=torch.optim.SGD(model2.parameters(), lr=learning_rate)
```


```python
print(list(model2.parameters())[0].size()) # output channels, filters, kernel height, kernel width
print(list(model2.parameters())[1].size())
print(list(model2.parameters())[2].size())
print(list(model2.parameters())[3].size())
print(list(model2.parameters())[4].size())
print(list(model2.parameters())[5].size())
```

    torch.Size([16, 1, 5, 5])
    torch.Size([16])
    torch.Size([32, 16, 5, 5])
    torch.Size([32])
    torch.Size([10, 1568])
    torch.Size([10])



```python
iter=0
for epoch in range(int(num_epochs)): # epoches=5
    for images, labels in train_gen: # len(train_gen=600)
        # convert inputs to Variables
        if iter==1:
            print(images.size()) # (100, 1, 28, 28)
        images=Variable(images)
        labels=Variable(labels)
         
        # clear grad buffers
        optimizer.zero_grad()
        
        # get outputs with inputs
        outputs=model2(images)
        
        # loss
        loss=criterion(outputs, labels)
        
        # backward gradients
        loss.backward()
        
        # updating parameters
        optimizer.step()
        
        # 1 iter: forward + backward pass
        iter+=1
        
        if iter % 500==0:
            correct=0
            total=0
            for images, labels in test_gen:
                images=Variable(images)
                outputs=model2(images)
                # images.data.size(): (100, 784)
                # outputs.data.size(): (100, 10)
                _, predicted=torch.max(outputs.data, 1)
                
                total+=labels.size(0)
                
                correct+=(predicted==labels).sum()
            accuracy=correct/total*100
            print("Iteration: {}, Loss: {}, Accurracy: {}".format(iter, loss.data[0], accuracy))
```

    torch.Size([100, 1, 28, 28])
    Iteration: 500, Loss: 0.535190999507904, Accurracy: 83.67999999999999
    Iteration: 1000, Loss: 0.33017414808273315, Accurracy: 88.98
    Iteration: 1500, Loss: 0.3632943332195282, Accurracy: 90.38000000000001
    Iteration: 2000, Loss: 0.351043164730072, Accurracy: 91.4
    Iteration: 2500, Loss: 0.23424170911312103, Accurracy: 92.24
    Iteration: 3000, Loss: 0.44090530276298523, Accurracy: 93.01


## Conclusion: Avgpooling accuracy < Maxpooling accuracy

### another cnn: with valid padding (means no padding)


```python
class CNNModel3(nn.Module):
    def __init__(self):
        # inheritance
        super(CNNModel3,self).__init__()
        # conv 1
        self.conv1=nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1=nn.ReLU()
        # max pool 1
        self.maxpool1=nn.MaxPool2d(kernel_size=2)
        # conv 2
        self.conv2=nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1,padding=0)
        self.relu2=nn.ReLU()
        # max pool 2
        self.maxpool2=nn.MaxPool2d(kernel_size=2)
        
        # fc1
        # 1. (28-5)/1+1
        # 2. 24/2
        # 3. (12-5)/1+1
        # 4. 8/2
        self.fc1=nn.Linear(32*4*4, 10)
        
    def forward(self, inputs):
        # conv 1
        net=self.conv1(inputs)
        net=self.relu1(net)
        # max pool1
        net=self.maxpool1(net)
        # conv 2
        net=self.conv2(net)
        net=self.relu2(net)
        # max pool2
        out=self.maxpool2(net)
        
        # resize
        out=out.view(out.size(0), -1)
        out=self.fc1(out)
        return out
```


```python
model3=CNNModel3()
```


```python
criterion=nn.CrossEntropyLoss()
```


```python
learning_rate=0.01
optimizer=torch.optim.SGD(model2.parameters(), lr=learning_rate)
```


```python
print(list(model3.parameters())[0].size()) # output channels, filters, kernel height, kernel width
print(list(model3.parameters())[1].size())
print(list(model3.parameters())[2].size())
print(list(model3.parameters())[3].size())
print(list(model3.parameters())[4].size())
print(list(model3.parameters())[5].size())
```

    torch.Size([16, 1, 5, 5])
    torch.Size([16])
    torch.Size([32, 16, 5, 5])
    torch.Size([32])
    torch.Size([10, 512])
    torch.Size([10])



```python
iter=0
for epoch in range(int(num_epochs)): # epoches=5
    for images, labels in train_gen: # len(train_gen=600)
        # convert inputs to Variables
        if iter==1:
            print(images.size()) # (100, 1, 28, 28)
        images=Variable(images)
        labels=Variable(labels)
         
        # clear grad buffers
        optimizer.zero_grad()
        
        # get outputs with inputs
        outputs=model3(images)
        
        # loss
        loss=criterion(outputs, labels)
        
        # backward gradients
        loss.backward()
        
        # updating parameters
        optimizer.step()
        
        # 1 iter: forward + backward pass
        iter+=1
        
        if iter % 500==0:
            correct=0
            total=0
            for images, labels in test_gen:
                images=Variable(images)
                outputs=model2(images)
                # images.data.size(): (100, 784)
                # outputs.data.size(): (100, 10)
                _, predicted=torch.max(outputs.data, 1)
                
                total+=labels.size(0)
                
                correct+=(predicted==labels).sum()
            accuracy=correct/total*100
            print("Iteration: {}, Loss: {}, Accurracy: {}".format(iter, loss.data[0], accuracy))
```

    torch.Size([100, 1, 28, 28])
    Iteration: 500, Loss: 2.310438394546509, Accurracy: 93.01
    Iteration: 1000, Loss: 2.312199592590332, Accurracy: 93.01
    Iteration: 1500, Loss: 2.3113796710968018, Accurracy: 93.01
    Iteration: 2000, Loss: 2.319981336593628, Accurracy: 93.01
    Iteration: 2500, Loss: 2.306147813796997, Accurracy: 93.01
    Iteration: 3000, Loss: 2.300149917602539, Accurracy: 93.01


## Conclusion: accuracy between model and model2
