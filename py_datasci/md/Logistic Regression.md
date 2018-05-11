
## Logistic Regression

$$h_{\theta}(x)=g(\theta^T x)$$

$$\text{where    }g(z)=\frac{1}{1+e^{-z}}$$

$$\text{thus   } h_{\theta}(x)=\frac{1}{1+e^{-\theta^Tx}}$$



### Interpretation of hypothesis output

$$h_{\theta}(x)=P(y=1 \mid x;\theta)$$

$$\text{probability that } y=1, \text{given } x, \text{parameterized by } \theta$$

$$\text{predict  }y=1 \text{  if  } h_{\theta}(x) \geq 0.5, \text{      which means  } g(\theta^T x) \geq 0.5, \text{  so  } \theta^T x \geq 0$$
$$\text{predict  }y=0 \text{  if  } h_{\theta}(x) \leq 0.5  \text{      which means  } g(\theta^T x) \leq 0.5, \text{  so  } \theta^T x \leq 0$$


### cost function



![](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images/sub1.png)

#### if y=1
![](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images/22917.png)

![](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images/22918.png)


$$Cost=0 \text{  if  } y=1, h_{\theta}(x)=1, h_{\theta}(x) \rightarrow 0 \text{   Cost   } \rightarrow \infty$$

#### if y=0

![](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images/22919.png)

![](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images/22920.png)


![](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images/22915.png)

![](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images/22916.png)

### Cross-entropy (a more compact way to present the previous cost function)

$$Cost(h_{\theta}(x), y)=- \log (h_{\theta}(x)^{y} (1-h_{\theta}(x))^{1-y})= -y \log (h_{\theta}(x))- (1-y) \log (1-h_{\theta}(x))$$


__thus__


![](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images/sub2.png)

$$J(\theta)= \frac{1}{m} \sum^m_{i=1} Cost(h_{\theta}(x^i), y)=\frac{1}{m} \sum^m_{i=1} - \log (h_{\theta}(x^i)^{y^i} (1-h_{\theta}(x^i))^{1-y^i})$$

$$ = \frac{1}{m} \sum^m_{i=1} -y^i \log (h_{\theta}(x^i))- (1-y^i) \log (1-h_{\theta}(x^i))$$

$$ \text{To fit parameters   } \theta$$

$$\min_{\theta} J(\theta)$$

$$\text{To make a prediction given new x:   output }h_{\theta}(x)=\frac{1}{1+e^{-\theta ^Tx}}$$


__Gradient Descent__


![](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images/sub3.png)


![](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images/22921.png)

$$h_{\theta}^i(x)=P(y=i \mid x;\theta) \: (i=1,2,3)$$

$$\text{Train a logistic regression classifier  } h_{\theta}^i(x) \text{    for each class i to predict the prob that  } y=i$$

$$\text{On a new input to-be tested x,to make a prediction, pick the class i which maximizes   } \max_{i} h_{\theta}^i(x)$$


## Regularization for Linear Regression

#### Overfitting: if we have too many features, the learned hypothesis may fit the training set too well, but fail to generalize to new examples:

$$J(\theta)=\frac{1}{2m} \sum^m_{i=1} (h_{\theta}(x^i)-y^i)^2 \approx 0$$

![](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images/22922.png)

> Solutions?

- Reduce number of features:
    - feature engineering
    - model selection algorithm
- Regularization
    - keep all features, but reduce magnitudes of params (in our case above: theta)
        - work well when there're lots of features, each contributes approximately equally to predicting y 

__After regularization__:



![](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images/sub4.png)

- Notes:

$$\text{if } \lambda \text{   set to an extremely large value, say } \lambda = 10^{10} \text{ , then it tends to underfit, vice versa}$$


__GD__:


![](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images/sub5.png)

$$\theta_j := \theta_j (1- \alpha \frac{\lambda}{m}) - \frac{\alpha}{m} \sum ^m_{i=1}  ( h_{\theta} (x^i) - y^i  ) x^i_j$$

__Normal Equation__:

![](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images/sub6.png)

### Attention!!

![](https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images/sub7.png)


## Regularization for Logistic Regression

$$ J(\theta)= -  [ \frac{1}{m} \sum^m_{i=1} y^i \log (h_{\theta}(x^i))+ (1-y^i) \log (1-h_{\theta}(x^i))  ] + \frac{\lambda}{2m} \sum^{n}_{j=1} \theta_j^2$$



$$\text{    Repeat   }  \{ \theta_j := \theta_j - \alpha  ( \frac{1}{m} \sum^m_{i=1} (h_{\theta}(x^i)-y^i) x_j^i - \frac{\lambda}{m} \theta_j  )  \}$$

$$\text{the difference from above is: for here   } h_{\theta}(x)=\frac{1}{1+e^{-\theta^Tx}}$$


$$\frac{\partial}{\partial \theta_0} J(\theta)= \frac{1}{m} \sum^m_{i=1} (h_{\theta}(x^i)-y^i) x_0^i$$

$$\frac{\partial}{\partial \theta_1} J(\theta)= \frac{1}{m} \sum^m_{i=1} (h_{\theta}(x^i)-y^i) x_1^i - \frac{\lambda}{m} \theta_1$$


$$\frac{\partial}{\partial \theta_2} J(\theta)= \frac{1}{m} \sum^m_{i=1} (h_{\theta}(x^i)-y^i) x_2^i - \frac{\lambda}{m} \theta_2$$

$$\text{.....}$$


$$\frac{\partial}{\partial \theta_n} J(\theta)= \frac{1}{m} \sum^m_{i=1} (h_{\theta}(x^i)-y^i) x_n^i - \frac{\lambda}{m} \theta_n$$
