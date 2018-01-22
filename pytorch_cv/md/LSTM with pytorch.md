

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

## make dataset iterable
- total: 60000
- batch_size: 100
- iterations: 3000
    - 1 iter: one minibatch forward and backward pass
- epochs
    - 1 epoch: running through the whole dataset once
    - epochs = $$ iter \div \frac{total data}{batchsize}= \frac{iter}{batches}=3000 \div \frac{60000}{100}=5 $$


```python
iter=6500
batch_size=100
num_epochs=iter/(len(train_dataset)/batch_size)
num_epochs
```




   20.0



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
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()

        self.hidden_dim=hidden_dim

        self.layer_dim=layer_dim

        self.lstm=nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)

        self.fc=nn.Linear(hidden_dim, output_dim)

    def forward(self, inputs):
        # initialize hidden state with zeros
        # (layer_dim, batch_size, hidden_dim)
        h0=Variable(torch.zeros(self.layer_dim, inputs.size(0), self.hidden_dim))
        c0=Variable(torch.zeros(self.layer_dim, inputs.size(0), self.hidden_dim))

        # one time step
        out, (hn, cn)=self.lstm(inputs, (h0, c0))

        # index hidden state of last time step
        # out.size() --> 100, 28, 100
        # out[:, -1, :] --> 100, 100
        out=self.fc(out[:, -1, :])
        return out

```


```python
input_dim=28
hidden_dim=100
layer_dim=2
output_dim=10
```


```python
model=LSTMModel(input_dim, hidden_dim, layer_dim, output_dim)
```


```python
criterion=nn.CrossEntropyLoss()
```

### optimizer function
$$\theta = \theta - \eta \cdot \triangledown_{\theta}$$


```python
learning_rate=0.05 # here set lr=0.01 or 0.1 wouldn't work
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)
```


```python
print(list(model.parameters())[0].shape) # input --> hidden
print(list(model.parameters())[2].shape) # input --> hidden bias
print(list(model.parameters())[1].shape) # hidden --> hidden
print(list(model.parameters())[3].shape) # hidden --> hidden bias
print(list(model.parameters())[4].shape) # hidden --> output
print(list(model.parameters())[5].shape) # hidden --> output bias
```

    torch.Size([400, 28])
    torch.Size([400])
    torch.Size([400, 100])
    torch.Size([400])
    torch.Size([400, 100])
    torch.Size([400, 100])


## train model



```python
seq_dim=28

iter=0
for epoch in range(int(num_epochs)): # epoches=5
    for images, labels in train_gen: # len(train_gen=600)

        # convert inputs to Variables
        if iter==1:
            print(images.size()) # (100,1, 28, 28)
            print(images.view(-1, seq_dim, input_dim).size()) # (100, 784)

        images=Variable(images.view(-1, seq_dim, input_dim))
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
                images=Variable(images.view(-1, seq_dim,input_dim))
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
    torch.Size([100, 28, 28])
    Iteration: 500, Loss: 2.3008172512054443, Accurracy: 11.35
    Iteration: 1000, Loss: 2.2902607917785645, Accurracy: 11.66
    Iteration: 1500, Loss: 2.1921539306640625, Accurracy: 19.88
    Iteration: 2000, Loss: 1.5763664245605469, Accurracy: 47.17
    Iteration: 2500, Loss: 0.7911041975021362, Accurracy: 70.43
    Iteration: 3000, Loss: 0.5378015637397766, Accurracy: 83.28
    Iteration: 3500, Loss: 0.38083571195602417, Accurracy: 91.24
    Iteration: 4000, Loss: 0.1958393156528473, Accurracy: 93.47
    Iteration: 4500, Loss: 0.31514883041381836, Accurracy: 94.91000000000001
    Iteration: 5000, Loss: 0.13328437507152557, Accurracy: 95.34
    Iteration: 5500, Loss: 0.20370596647262573, Accurracy: 95.45
    Iteration: 6000, Loss: 0.13825298845767975, Accurracy: 96.05
    Iteration: 6500, Loss: 0.2114141434431076, Accurracy: 96.82

