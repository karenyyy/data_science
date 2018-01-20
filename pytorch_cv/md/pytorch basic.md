

```python
import torch
import matplotlib.pyplot as plt
```


```python
array=[[1,2], [3,4]]
torch.Tensor(array)
```




  
    1  2
    3  4
    [torch.FloatTensor of size 2x2]




```python
if torch.cuda.is_available():
    torch.rand(2,2).cuda()
```


```python
import numpy as np

np_array=np.ones((2,2), dtype=np.uint8)
print(torch.from_numpy(np_array))
```

    
     1  1
     1  1
    [torch.ByteTensor of size 2x2]
    



```python
torch_tensor=torch.ones(2,2)
torch_tensor.numpy()
```




   array([[ 1.,  1.],
           [ 1.,  1.]], dtype=float32)




```python
torch.Tensor([1,2,3,4,5,6,7,8,9,10]).std(dim=0)
```




    
   3.0277
   [torch.FloatTensor of size 1]



## Variables and gradients


```python
from torch.autograd import Variable
a=Variable(torch.ones(2,2), requires_grad=True)
b=Variable(torch.rand(2,2), requires_grad=True)
torch.mul(a,b)
```




   Variable containing:
   
     0.5322  0.3271
     0.5562  0.6388
    [torch.FloatTensor of size 2x2]




```python
x=Variable(torch.ones(1,1), requires_grad=True)
y=5*(x+1)**2
o=(1/2)*torch.sum(y)
o
o.backward()
x.grad
```




   Variable containing:
     10
    [torch.FloatTensor of size 1x1]




```python
n=50
x=np.random.randn(n)
y=x*np.random.randn(n)

colors=np.random.rand(n)

plt.plot(np.unique(x),np.poly1d(np.polyfit(x,y,1))(np.unique(x)))
plt.scatter(x,y,c=colors,alpha=0.5)
plt.show()
```


![png](output_9_0.png)


## Build model


```python
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
```


```python
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegressionModel, self).__init__()
        self.linear=nn.Linear(in_features=input_size, out_features=output_size)
    
    def forward(self, x):
        return self.linear(x)
```


```python
input_dim=1
output_dim=1

model=LinearRegressionModel(input_dim, output_dim)
```


```python
criterion=nn.MSELoss()
```


```python
learning_rate=0.01
optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)
```


```python
epoches=100

x=[i for i in range(10)]
x_train=np.array(x, dtype=np.float32)
x_train=x_train.reshape(-1, 1) # make sure it only got one column

y=[2*i+1 for i in range(10)]
y_train=np.array(y, dtype=np.float32)
y_train=y_train.reshape(-1, 1)


for epoch in range(epoches):
  
    inputs=Variable(torch.from_numpy(x_train))
    labels=Variable(torch.from_numpy(y_train))
    
    # clear gradients and parameters
    optimizer.zero_grad()
    outputs=model(inputs)
    
    loss=criterion(outputs, labels)
    loss.backward()
    
    # updating parameters
    optimizer.step()
    
    print("epoch {}, loss {}".format(epoch, loss.data[0]))
```

    epoch 0, loss 106.28778076171875
    epoch 1, loss 18.60418701171875
    epoch 2, loss 3.453016757965088
    epoch 3, loss 0.832769513130188
    epoch 4, loss 0.3774281442165375
    epoch 5, loss 0.2961316406726837
    epoch 6, loss 0.2794860005378723
    epoch 7, loss 0.27403855323791504
    epoch 8, loss 0.2705548107624054
    epoch 9, loss 0.267439067363739
    epoch 10, loss 0.2644151449203491
    epoch 11, loss 0.2614351212978363
    epoch 12, loss 0.2584902048110962
    epoch 13, loss 0.2555789649486542
    epoch 14, loss 0.2527005076408386
    epoch 15, loss 0.24985432624816895
    epoch 16, loss 0.24704034626483917
    epoch 17, loss 0.24425789713859558
    epoch 18, loss 0.2415069043636322
    epoch 19, loss 0.23878693580627441
    epoch 20, loss 0.23609760403633118
    epoch 21, loss 0.23343853652477264
    epoch 22, loss 0.23080937564373016
    epoch 23, loss 0.22820977866649628
    epoch 24, loss 0.22563953697681427
    epoch 25, loss 0.2230982780456543
    epoch 26, loss 0.22058573365211487
    epoch 27, loss 0.21810121834278107
    epoch 28, loss 0.2156449556350708
    epoch 29, loss 0.21321602165699005
    epoch 30, loss 0.21081475913524628
    epoch 31, loss 0.20844046771526337
    epoch 32, loss 0.20609283447265625
    epoch 33, loss 0.203771710395813
    epoch 34, loss 0.20147661864757538
    epoch 35, loss 0.19920754432678223
    epoch 36, loss 0.19696375727653503
    epoch 37, loss 0.1947455108165741
    epoch 38, loss 0.19255222380161285
    epoch 39, loss 0.19038346409797668
    epoch 40, loss 0.18823926150798798
    epoch 41, loss 0.1861191987991333
    epoch 42, loss 0.18402303755283356
    epoch 43, loss 0.18195047974586487
    epoch 44, loss 0.17990124225616455
    epoch 45, loss 0.17787505686283112
    epoch 46, loss 0.17587170004844666
    epoch 47, loss 0.17389102280139923
    epoch 48, loss 0.17193244397640228
    epoch 49, loss 0.16999609768390656
    epoch 50, loss 0.16808141767978668
    epoch 51, loss 0.16618840396404266
    epoch 52, loss 0.1643165647983551
    epoch 53, loss 0.16246607899665833
    epoch 54, loss 0.1606362760066986
    epoch 55, loss 0.1588270217180252
    epoch 56, loss 0.15703821182250977
    epoch 57, loss 0.1552695780992508
    epoch 58, loss 0.15352068841457367
    epoch 59, loss 0.15179160237312317
    epoch 60, loss 0.1500820815563202
    epoch 61, loss 0.14839184284210205
    epoch 62, loss 0.14672057330608368
    epoch 63, loss 0.14506812393665314
    epoch 64, loss 0.14343427121639252
    epoch 65, loss 0.1418187916278839
    epoch 66, loss 0.14022156596183777
    epoch 67, loss 0.1386423408985138
    epoch 68, loss 0.1370808184146881
    epoch 69, loss 0.13553689420223236
    epoch 70, loss 0.13401038944721222
    epoch 71, loss 0.1325010359287262
    epoch 72, loss 0.13100871443748474
    epoch 73, loss 0.12953321635723114
    epoch 74, loss 0.12807440757751465
    epoch 75, loss 0.12663185596466064
    epoch 76, loss 0.12520566582679749
    epoch 77, loss 0.12379548698663712
    epoch 78, loss 0.1224011555314064
    epoch 79, loss 0.12102259695529938
    epoch 80, loss 0.11965961754322052
    epoch 81, loss 0.11831194162368774
    epoch 82, loss 0.11697951704263687
    epoch 83, loss 0.11566201597452164
    epoch 84, loss 0.11435934156179428
    epoch 85, loss 0.11307138204574585
    epoch 86, loss 0.11179782450199127
    epoch 87, loss 0.11053870618343353
    epoch 88, loss 0.10929372161626816
    epoch 89, loss 0.10806278139352798
    epoch 90, loss 0.10684569180011749
    epoch 91, loss 0.10564231872558594
    epoch 92, loss 0.10445252805948257
    epoch 93, loss 0.10327613353729248
    epoch 94, loss 0.10211290419101715
    epoch 95, loss 0.10096291452646255
    epoch 96, loss 0.09982583671808243
    epoch 97, loss 0.09870152175426483
    epoch 98, loss 0.09758985787630081
    epoch 99, loss 0.09649074077606201



```python
estimated=model(Variable(torch.from_numpy(x_train))).data.numpy()
estimated
```




    array([[  0.42598221],
           [  2.51752353],
           [  4.60906458],
           [  6.70060587],
           [  8.79214764],
           [ 10.88368893],
           [ 12.97523022],
           [ 15.06677151],
           [ 17.15831184],
           [ 19.24985313]], dtype=float32)




```python
y_train
```




    array([[  1.],
           [  3.],
           [  5.],
           [  7.],
           [  9.],
           [ 11.],
           [ 13.],
           [ 15.],
           [ 17.],
           [ 19.]], dtype=float32)



### plot the result


```python
# clear the canvas
plt.clf()

estimated=model(Variable(torch.from_numpy(x_train))).data.numpy()

plt.plot(x_train, y_train, "go", label="True data", alpha=0.5) # "go": green, o for points
plt.plot(x_train, estimated, "--", label="Predictions", alpha=0.5)
plt.legend(loc="best") # auto locate the most appropriate location on graph
plt.show()
```


![png](output_20_0.png)


### save the model parameters


```python
save_model=False
if save_model is True:
    torch.save(model.state_dict(), "basic_model.pkl")
```
