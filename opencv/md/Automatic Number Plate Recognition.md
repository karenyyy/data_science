

```python
import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os

%matplotlib inline

IMG = os.path.join("../","images","new_scene.png")

img = cv2.imread(IMG, cv2.IMREAD_COLOR)

fig = plt.figure(figsize=(50,50))

fig.add_subplot(9,1,1).imshow(img)

# RGB to Gray scale conversion
img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
fig.add_subplot(9,1,2).imshow(img_gray)
plt.title('gray scale')

# Noise removal with iterative bilateral filter(removes noise while preserving edges)
noise_removal = cv2.bilateralFilter(img_gray,9,75,75)
fig.add_subplot(9,1,3).imshow(noise_removal)
plt.title('bilateral filter')

# Histogram equalisation for better results
equal_histogram = cv2.equalizeHist(noise_removal)
fig.add_subplot(9,1,4).imshow(equal_histogram)
plt.title('equalizeHist')

# Morphological opening with a rectangular structure element
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
morph_image = cv2.morphologyEx(equal_histogram,cv2.MORPH_OPEN,kernel,iterations=15)
fig.add_subplot(9,1,5).imshow(morph_image)
plt.title('morphological')

# Image subtraction(Subtracting the Morphed image from the histogram equalised Image)
sub_morp_image = cv2.subtract(equal_histogram,morph_image)
fig.add_subplot(9,1,6).imshow(sub_morp_image)
plt.title('subtract morphed img from equalizedHist')

# Thresholding the image
ret,thresh_image = cv2.threshold(sub_morp_image,0,255,cv2.THRESH_OTSU)
fig.add_subplot(9,1,7).imshow(thresh_image)
plt.title('threshold img')

# Applying Canny Edge detection
canny_image = cv2.Canny(thresh_image,250,255)
fig.add_subplot(9,1,8).imshow(canny_image)
plt.title('canny edge detection')

# dilation to strengthen the edges
kernel = np.ones((3,3), np.uint8)
dilated_image = cv2.dilate(canny_image,kernel,iterations=1)
fig.add_subplot(9,1,9).imshow(dilated_image)
plt.title('dilation')

plt.show()
```


![png](output_0_0.png)



```python
# Finding Contours in the image based on edges
new,contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10]

final = img
screenCnt = None

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # Approximating with 6% error
    screenCnt = approx
    final = cv2.drawContours(final, [approx], -1, (0, 255, 0), 3) 
    
fig = plt.figure(figsize=(50, 50))
fig.add_subplot(3,1,1).imshow(final)


# Masking the part other than the number plate
mask = np.zeros(img_gray.shape,np.uint8)
new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
new_image = cv2.bitwise_and(img,img,mask=mask)
fig.add_subplot(3,1,2).imshow(new_image)


# Histogram equal for enhancing the number plate for further processing
y,cr,cb = cv2.split(cv2.cvtColor(new_image,cv2.COLOR_RGB2YCrCb))
# Converting the image to YCrCb model and splitting the 3 channels
y = cv2.equalizeHist(y)
# Applying histogram equalisation
final_image = cv2.cvtColor(cv2.merge([y,cr,cb]),cv2.COLOR_YCrCb2RGB)

fig.add_subplot(3,1,3).imshow(final_image)
plt.show()
```


![png](output_1_0.png)


### Code to automatically segment numbers from numberplates by finding contours



```python
IMG = os.path.join("../","images","new_plate.png")

img = cv2.imread(IMG)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
noise_removal = cv2.bilateralFilter(gray_img,9,75,75)
ret,thresh_image = cv2.threshold(noise_removal,70,255,cv2.THRESH_BINARY_INV)

new,contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours= sorted(contours, key = cv2.contourArea, reverse = True)

for cnt in contours:
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

plt.imshow(img)
plt.show()
```


![png](output_3_0.png)



```python
# randomly generate number plates

from random import randint
 
def gen_plate_number(noc=7):    
    
    CANDIDATES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    NO_OF_CHARS = noc                                   
    
    plate = ""
    for ch in range(NO_OF_CHARS):
        num = randint(0,35)
        plate += CANDIDATES[num]
        
    return plate

print(gen_plate_number())
```

    M91LYOP



```python
import pickle
import requests
from bs4 import BeautifulSoup
from PIL import Image
import tensorflow as tf
%matplotlib inline 

CHARACTERS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
N_CLASSES = 36
```


```python
def read_dataset(fname="number_plates_new.pickle",N=None):
    f = open(fname,'rb')
    X_data,Y_label = pickle.load(f)
    if not N is None:
        return X_data[:N],Y_label[:N]
    else:
        return X_data, Y_label
```


```python
html = requests.get("http://acme.com/licensemaker/licensemaker.cgi", 
            params={
                "state" : "California",
                "text"  : 123456, 
                "plate" : "1998",
                "r"     : "1014181821"
            }).text
    
soup = BeautifulSoup(html,"html5lib")
soup('img')
```




    [<img alt="atom" class="logo" height="86" src="/resources/images/atom_ani.gif" width="86"/>,
     <img alt="license maker" height="50" src="licensemaker.jpg" width="228"/>,
     <img alt="license" src="licenses/license_20180213175935_23487.jpg"/>,
     <img alt="valid HTML" border="0" class="logo" height="31" src="/resources/images/valid-html401-gold.png" width="88"/>,
     <img alt="email" class="mailto" src="/mailto/wa.gif"/>]




```python
def generate_plate(plate_num):
    html = requests.get("http://acme.com/licensemaker/licensemaker.cgi", 
            params={
                "state" : "California",
                "text"  : plate_num, 
                "plate" : "1998",
                "r"     : "1014181821"
            }).text
    
    soup = BeautifulSoup(html,"html5lib")
    
    URL_DOM = "http://acme.com/licensemaker/"
    URL_IMG = soup('img')[2]['src']
    URL = URL_DOM + URL_IMG
    
    # Fetching the image from the web 
    response = requests.get(URL)
    if response.status_code == 200: # no server error
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        return np.array(cv2.imdecode(img_array, cv2.IMREAD_COLOR)),plate_num
    
    # if server error
    return None,plate_num
```


```python
def contour_center(cnt):
    
    # finding the moments of the contour
    M = cv2.moments(cnt)
    
    if M["m00"] != 0:
        # center x
        cX = int(M["m10"] / M["m00"])
    else:
        cX = 0
    
    return cX
```


```python
def one_hot_encoded(label_lst,classes=N_CLASSES):
    return list(np.eye(classes)[label_lst])
```


```python
def get_code(ch):
    OFFSET = 10
    if ch.isdigit():
        return ord(ch) - ord('0')
    else:
        return OFFSET + (ord(ch) - ord('A'))
```


```python
def get_img_label(img,plate_num):
    
    x_list, y_list = [], []
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    noise_removal = cv2.bilateralFilter(img,9,75,75)
    
    ret,thresh_image = cv2.threshold(noise_removal,70,255,cv2.THRESH_BINARY_INV)

    new,contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # contours sorted by their centers
    contours= sorted(contours, key = contour_center)[:7]
    
    for idx,cnt in enumerate(contours):
        
        x,y,w,h = cv2.boundingRect(cnt)
        
        part = img[y : y + h, x : x + w]
        
        part = cv2.resize(part,(40,18))
        
        _,part = cv2.threshold(part,128,255,cv2.THRESH_BINARY_INV)
        
        part = cv2.normalize(part, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        
        x_list.append(list(part))
        
        y_list.append(one_hot_encoded(get_code(num[idx])))
    
    return x_list, y_list
```


```python
def gen_training_dataset(runs,fname="number_plates_new.pickle"):
    X_data, Y_data = [], []
    
    if os.path.exists(fname):
        with open(fname,'rb') as rfp: 
            X_data, Y_data = ( list(data) for data in pickle.load(rfp) )
        rfp.close()
        
    print(len(X_data), len(Y_data))
    
    for idx in range(runs):
        
        # generating a new number
        num = gen_plate_number()
        img, num = generate_plate(num)
        
        x_list,y_list = get_img_label(img,num)
        
        X_data.extend(x_list)
        Y_data.extend(y_list)
    
    f = open(fname, 'wb')   
    pickle.dump((np.array(X_data), np.array(Y_data)), f)
    f.close()
```


```python
def one_hot_decode(lst):
    return np.argmax(lst)
```


```python
def get_char(num):
    OFFSET = 10
    if num < 10:
        return chr(ord('0') + num) # converting the number to char 
    else:
        return chr(ord('A') + num - OFFSET)

```


```python
def display_plate_numbers(img,num):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise_removal = cv2.bilateralFilter(gray_img,9,75,75)
    ret,thresh_image = cv2.threshold(noise_removal,70,255,cv2.THRESH_BINARY_INV)

    new,contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # contours sorted by their centers
    contours= sorted(contours, key = contour_center)[:7]
    
    fig = plt.figure()
    
    for idx,cnt in enumerate(contours):
        x,y,w,h = cv2.boundingRect(cnt)
        part = img[y : y + h, x : x + w]
        part = cv2.cvtColor(part,cv2.COLOR_BGR2GRAY)
        _,part = cv2.threshold(part,128,255,cv2.THRESH_BINARY_INV)
        fig.add_subplot(4,2,idx+1).imshow(part)
    plt.show()


def display_plate(img,num):
    print("Plate Label : ", num )
    plt.imshow(img)
    plt.show()

def display_plate_detected(img):
    
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise_removal = cv2.bilateralFilter(gray_img,9,75,75)
    ret,thresh_image = cv2.threshold(noise_removal,70,255,cv2.THRESH_BINARY)

    new,contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours= sorted(contours, key = cv2.contourArea, reverse = True)

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

    plt.gray()
    plt.imshow(img)
    plt.show()
```


```python
img,num = generate_plate(gen_plate_number())
display_plate(img,num)
display_plate_numbers(img,num)
```

    Plate Label :  BAK90R6



![png](output_17_1.png)



![png](output_17_2.png)



```python
# import os
# fname = "number_plates_new.pickle"
# gen_training_dataset(100, fname)
```


```python
# Reading Dataset
X_data,Y_label = read_dataset(fname)
print(X_data.shape, Y_label.shape)
```

    (4900, 18, 40) (4900, 36)



```python
fname = "number_plates_new.pickle"
X_data,Y_label = read_dataset(fname,5)

for image, label in zip(X_data,Y_label):
    print(get_char(one_hot_decode(label)))
    plt.figure(figsize=(2,2))
    plt.imshow(image)
    plt.show()
```

    Q



![png](output_20_1.png)


    B



![png](output_20_3.png)


    Q



![png](output_20_5.png)


    E



![png](output_20_7.png)


    X



![png](output_20_9.png)



```python
X_data,Y_label = read_dataset()

def read_data_sets(num):
    
    cars = {}
    i = 0
    
    for image, label in zip(X_data, Y_label):
        if i < num:
            if not 'train' in cars:
                cars['train'] = {'images': [], 'labels': []}
            cars['train']['images'].append(image.flatten())
            cars['train']['labels'].append(label)
        else:
            if not 'test' in cars:
                cars['test'] = {'images': [], 'labels': []}
            cars['test']['images'].append(image.flatten())
            cars['test']['labels'].append(label)
        
        i+=1
        
    cars['train']['images'] = np.array(cars['train']['images'])
    cars['train']['labels'] = np.array(cars['train']['labels'])
    cars['test']['images']  = np.array(cars['test']['images'])
    cars['test']['labels']  = np.array(cars['test']['labels'])
        
    return cars


cars = read_data_sets(4850)

print(cars['train']['images'].shape)
print(cars['train']['labels'].shape)
print(cars['test']['images'].shape)
print(cars['test']['labels'].shape)
```

    (4850, 720)
    (4850, 36)
    (50, 720)
    (50, 36)



```python
#### Create the model
x = tf.placeholder(tf.float32, [None, 720])
W = tf.Variable(tf.zeros([720, 36]))
b = tf.Variable(tf.zeros([36]))
y = tf.matmul(x, W) + b

y_ = tf.placeholder(tf.float32, [None, 36])

cross_entropy = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

saver = tf.train.Saver()
    
# Train
batch_xs, batch_ys = cars['train']['images'], cars['train']['labels']

sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(sess.run(accuracy, feed_dict={x: cars['test']['images'],
                                          y_: cars['test']['labels']}))
```

    0.76



```python
directory="model/plate_model.ckpt"
if not os.path.exists(directory):
    os.makedirs(directory)
save_path = saver.save(sess, directory)
print("Model saved in file: %s" % save_path)
```

    Model saved in file: model/plate_model.ckpt



```python

def predictint(imvalue):
    """
    This function returns the predicted integer.
    The input is the pixel values from the imageprepare() function.
    """            
    prediction=tf.argmax(y,1)
    return prediction.eval(feed_dict={x: [imvalue]}, session=sess)

# Test images generation

X_data,Y_label = read_dataset(fname,10)

for image, label in zip(X_data,Y_label):
    imvalue = np.array(image).flatten()
    
    label = predictint(imvalue)
    print(get_char(label[0]))
    plt.figure(figsize=(2,2))
    print(image.shape)
    plt.imshow(image)
    plt.show()
```

    O
    (18, 40)



![png](output_24_1.png)


    B
    (18, 40)



![png](output_24_3.png)


    O
    (18, 40)



![png](output_24_5.png)


    E
    (18, 40)



![png](output_24_7.png)


    X
    (18, 40)



![png](output_24_9.png)


    G
    (18, 40)



![png](output_24_11.png)


    S
    (18, 40)



![png](output_24_13.png)


    4
    (18, 40)



![png](output_24_15.png)


    G
    (18, 40)



![png](output_24_17.png)


    E
    (18, 40)



![png](output_24_19.png)



```python
# Training the characters for MNIST

import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Create the model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.matmul(x, W) + b

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(
          tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
    
# Train
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Test trained model
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(mnist.test.images.shape, mnist.test.labels.shape)

print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                          y_: mnist.test.labels}))

directory="model/mnist_plate_model.ckpt"
if not os.path.exists(directory):
    os.makedirs(directory)
save_path = saver.save(sess, directory)
print("Model saved in file: %s" % save_path)
```

    Extracting MNIST_data/train-images-idx3-ubyte.gz
    Extracting MNIST_data/train-labels-idx1-ubyte.gz
    Extracting MNIST_data/t10k-images-idx3-ubyte.gz
    Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
    (10000, 784) (10000, 10)
    0.9144
    Model saved in file: model/mnist_plate_model.ckpt



```python
tf_keras = tf.contrib.keras
fig = plt.figure(figsize=(10,10))

(x_train, y_train), (x_test, y_test) = tf_keras.datasets.mnist.load_data(
    "mnist.npz")

def predictint2(imvalue):
    """
    This function returns the predicted integer.
    The input is the pixel values from the imageprepare() function.
    """
    prediction=tf.argmax(y,1)
    return prediction.eval(feed_dict={x: [imvalue]}, session=sess)

# Test images generation
X_data,Y_label = x_test[:10], y_test[:10]

for image, label in zip(X_data,Y_label):
    imvalue = np.array(image).flatten()
    label = predictint2(imvalue)
    print(get_char(label[0]))
    plt.figure(figsize=(2,2))
    image= cv2.resize(image, (28, 28))
    plt.imshow(image)
    plt.show()
```

    7



    <matplotlib.figure.Figure at 0x7f90251187b8>



![png](output_26_2.png)


    2



![png](output_26_4.png)


    1



![png](output_26_6.png)


    0



![png](output_26_8.png)


    4



![png](output_26_10.png)


    1



![png](output_26_12.png)


    4



![png](output_26_14.png)


    9



![png](output_26_16.png)


    6



![png](output_26_18.png)


    9



![png](output_26_20.png)



```python
def match_area(cnt):
    area = cv2.contourArea(cnt) 
    return area >= 1400 and area <= 2000
```


```python
def predict_number(img):
    
    detected_number = ''
     
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    equal_histogram = cv2.equalizeHist(img_gray)
    
    ret,thresh_image = cv2.threshold(equal_histogram,50,255,cv2.THRESH_BINARY_INV)
  
    new, contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print("found {} contours: ".format(len(contours)))
    contours= sorted(contours, key = contour_center)
      
    count = 0
    
    fig = plt.figure("Input Image",figsize=(10,10))
          
    for idx, cnt in enumerate(contours):
        
        p,q,w,h = cv2.boundingRect(cnt)
        
        aspect_ratio = h / w
            
        if aspect_ratio >= 2.1 and aspect_ratio <= 2.85:
            
            count += 1
        
            part = img_gray[q : q + h, p : p + w]
            
            cv2.rectangle(img,(p,q),(p+w,q+h),(0,255,0),1)
            
            # Resize when using custom model
            #part = cv2.resize(part,(40,18), interpolation=cv2.INTER_NEAREST)
            
            # Resize to MNIST model
            part = cv2.resize(part,(28,28), interpolation=cv2.INTER_NEAREST)
            
            _,part = cv2.threshold(part,128,255,cv2.THRESH_BINARY_INV)
            
            
            # lets do erosion
            kernel = np.ones((3,3),np.uint8)
            part = cv2.erode(part,kernel,iterations = 1)

            part = cv2.normalize(part, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            label = predictint2(part.flatten())
            label = get_char(label[0])
            
            detected_number += label

            fig.add_subplot(4,4,idx+1).imshow(part)
            
        plt.show()
    
    return detected_number
    
```


```python
def predict_number(img):
    
    detected_number = ''
     
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    equal_histogram = cv2.equalizeHist(img_gray)
    
    ret,thresh_image = cv2.threshold(equal_histogram,50,255,cv2.THRESH_BINARY_INV)
  
    new, contours, hierarchy = cv2.findContours(thresh_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print("found {} contours: ".format(len(contours)))
    contours= sorted(contours, key = contour_center)
      
    count = 0
    
    fig = plt.figure("Input Image",figsize=(10,10))
          
    for idx, cnt in enumerate(contours):
        
        p,q,w,h = cv2.boundingRect(cnt)
        
        aspect_ratio = h / w
            
        if aspect_ratio >= 2.1 and aspect_ratio <= 2.85:
            
            count += 1
        
            part = img_gray[q : q + h, p : p + w]
            
            cv2.rectangle(img,(p,q),(p+w,q+h),(0,255,0),1)
            
            # Resize when using custom model
            #part = cv2.resize(part,(40,18), interpolation=cv2.INTER_NEAREST)
            
            # Resize to MNIST model
            part = cv2.resize(part,(28,28), interpolation=cv2.INTER_NEAREST)
            
            _,part = cv2.threshold(part,128,255,cv2.THRESH_BINARY_INV)
            
            
            # lets do erosion
            kernel = np.ones((3,3),np.uint8)
            part = cv2.erode(part,kernel,iterations = 1)

            part = cv2.normalize(part, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

            label = predictint2(part.flatten())
            label = get_char(label[0])
            
            detected_number += label

            fig.add_subplot(4,4,idx+1).imshow(part)
    
    return detected_number
    
IMG = os.path.join("../","images","new_scene.png")

img = cv2.imread(IMG)

# Gray Image
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Histogram equalisation for better results
equal_histogram = cv2.equalizeHist(img_gray)

_, thres_image = cv2.threshold(equal_histogram, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

new, contours, hierarchy = cv2.findContours(thres_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

num_plates = 0


fig = plt.figure("Input Image",figsize=(10,10))

for cnt in contours:
    rect = cv2.minAreaRect(cnt)
    width = rect[1][0]
    height = rect[1][1]
    
    area = cv2.contourArea(cnt) 
    if area >= 1400 and area <= 1500:
            
        num_plates += 1
        
        p,q,w,h = cv2.boundingRect(cnt)
        
        part = img[q : q + h, p : p + w]
        
        number=predict_number(part)
        
        print(number)
        
        plt.imshow(part)

    
print("No of number plates detected : {0}".format(num_plates))

plt.show()

```

    found 16 contours: 
    5555555
    No of number plates detected : 1



![png](output_29_1.png)



```python
fig = plt.figure("Input Image",figsize=(10,10))
fig.add_subplot(1,1,1).imshow(img)

plt.show()
```


![png](output_30_0.png)

