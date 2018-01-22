
### Output: probability [0, 1] given input belonging to a class


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
len(train_dataset)
```




    60000




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


![png](https://raw.githubusercontent.com/karenyyy/data_science/master/pytorch_cv/images/output_7_0.png)



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
                                     batch_size=batch_size)
```


```python
len(train_gen)
```




    600



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


```python
import collections
isinstance(test_gen, collections.Iterable)
```




    True



## build the model


```python
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear=nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)
```


```python
input_dim=28*28
output_dim=10

model=LogisticRegressionModel(input_dim, output_dim)
```


```python
if torch.cuda.is_available():
    model.cuda()
```

### loss function
- CrossEntropyLoss
    - softmax
    - cross entropy


```python
criterion=nn.CrossEntropyLoss()
```

### optimizer function
$$\theta = \theta - \eta \cdot \triangledown_{\theta}$$


```python
learning_rate=0.001
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)
```


```python
print(list(model.parameters()))
```

    [Parameter containing:
     2.9148e-02  2.0584e-02 -1.8756e-02  ...  -7.5109e-03  1.5949e-03 -1.7543e-02
    -2.1806e-02  3.0784e-03  3.4229e-02  ...  -3.3077e-03  8.7504e-03  2.8028e-02
     2.8973e-02  1.4081e-02  2.3112e-02  ...   1.0536e-02 -3.0887e-03 -3.1467e-02
                    ...                   ⋱                   ...                
    -3.0432e-02 -7.1189e-03  2.9053e-02  ...  -1.4085e-02 -1.1206e-02  3.2586e-02
    -1.8221e-02 -4.3541e-03 -1.3614e-02  ...  -1.9895e-02  1.8769e-02 -1.2394e-02
    -4.9883e-03  2.3742e-02 -7.9224e-03  ...   3.1887e-02  8.5291e-03  1.7561e-02
    [torch.cuda.FloatTensor of size 10x784 (GPU 0)]
    , Parameter containing:
    -0.0411
     0.1093
    -0.0336
    -0.0077
     0.0263
     0.0664
     0.0044
     0.0746
    -0.1193
    -0.0417
    [torch.cuda.FloatTensor of size 10 (GPU 0)]
    ]



```python
# FC 1 parameters
list(model.parameters())[0]
```




    Parameter containing:
     2.9148e-02  2.0584e-02 -1.8756e-02  ...  -7.5109e-03  1.5949e-03 -1.7543e-02
    -2.1806e-02  3.0784e-03  3.4229e-02  ...  -3.3077e-03  8.7504e-03  2.8028e-02
     2.8973e-02  1.4081e-02  2.3112e-02  ...   1.0536e-02 -3.0887e-03 -3.1467e-02
                    ...                   ⋱                   ...                
    -3.0432e-02 -7.1189e-03  2.9053e-02  ...  -1.4085e-02 -1.1206e-02  3.2586e-02
    -1.8221e-02 -4.3541e-03 -1.3614e-02  ...  -1.9895e-02  1.8769e-02 -1.2394e-02
    -4.9883e-03  2.3742e-02 -7.9224e-03  ...   3.1887e-02  8.5291e-03  1.7561e-02
    [torch.cuda.FloatTensor of size 10x784 (GPU 0)]




```python
# FC 1 bias parameters
list(model.parameters())[1]
```




    Parameter containing:
    -0.0411
     0.1093
    -0.0336
    -0.0077
     0.0263
     0.0664
     0.0044
     0.0746
    -0.1193
    -0.0417
    [torch.cuda.FloatTensor of size 10 (GPU 0)]



<img src="https://raw.githubusercontent.com/karenyyy/data_science/master/pytorch_cv/images/1.png">

## train model
- convert inputs images and labels to Variables
- get output given inputs
- clear gradient buffers
- __get loss__
- __get gradients__ regarding to required parameters
- __update parameters__ using gradients
    - para=para-learning rate * grad
- __LOOP__


```python
iter=0
for epoch in range(int(num_epochs)): # epoches=5
    for images, labels in train_gen: # len(train_gen=600)
        
        # convert inputs to Variables
        images=Variable(images.view(-1, 28*28).cuda())
        labels=Variable(labels.cuda())
         
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
                images=Variable(images.view(-1, 28*28).cuda())
                outputs=model(images)
                # images.data.size(): (100, 784)
                # outputs.data.size(): (100, 10)
                _, predicted=torch.max(outputs.data, 1)
                
                total+=labels.size(0)
                
                correct+=(predicted.cuda()==labels.cuda()).sum()
            accuracy=correct/total*100
            print("Iteration: {}, Loss: {}, Accurracy: {}".format(iter, loss.data[0], accuracy))
```

    Iteration: 500, Loss: 0.6537198424339294, Accurracy: 87.59
    Iteration: 1000, Loss: 0.6380895376205444, Accurracy: 87.74
    Iteration: 1500, Loss: 0.738538920879364, Accurracy: 87.81
    Iteration: 2000, Loss: 0.5727197527885437, Accurracy: 87.76
    Iteration: 2500, Loss: 0.40449437499046326, Accurracy: 87.94999999999999
    Iteration: 3000, Loss: 0.5783911943435669, Accurracy: 88.01



```python
iter_test=0
for images, labels in test_gen:
    images=Variable(images.view(-1, 28*28).cuda())
    outputs=model(images)
    _,predicted=torch.max(outputs.data, 1)
    iter_test+=1
    if iter_test==1:
        print((predicted.cuda()==labels.cuda()).sum()/len(predicted)*100, "% testing accuracy")
```

    89.0 % testing accuracy

