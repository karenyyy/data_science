
### Image Features
- Image features include:
    - color features (RGB histogram)
    - texture features
    - visual similarity measure
    - semantic feature
    - color entropy

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

img = cv2.imread('../piedpiper.png')

cv2.imwrite('../compressed_pp.png', img, [cv2.IMWRITE_PNG_COMPRESSION])

```




    True




```python
img = cv2.imread('../piedpiper.png', cv2.IMREAD_COLOR)
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
y,u,v = cv2.split(yuv_img)

plt.imshow(gray_img)
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_1_1.png)



```python
plt.imshow(y)
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_2_1.png)



```python
plt.imshow(u)
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_3_1.png)



```python
plt.imshow(v)
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_4_1.png)



```python
g,b,r = cv2.split(img)
gbr_img = cv2.merge((g,b,r))
rbr_img = cv2.merge((r,b,r))
```


```python
plt.imshow(gbr_img)
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_6_1.png)



```python
plt.imshow(rbr_img)
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_7_1.png)


### Transformation

> How warping works?

Translation basically means that we are __shifting the image by adding/subtracting the x and y coordinates__. 

In order to do this, we need to create a __transformation matrix__:

![](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/matrix.png)

Here, the tx and ty values are the x and y translation values

the image will be moved by __x units to the right, and by y units downwards__. So once we create a matrix like this, we can use the function, __warpAffine__, to apply it to our image. 


__The third argument in warpAffine refers to the number of rows and columns in the resulting image.__

As follows, it passes __InterpolationFlags which defines combination of interpolation methods__.

Since the number of rows and columns is the same as the original image, __the resultant image is going to get cropped.__

The reason for this is we didn't have enough space in the output when we applied the translation matrix. To avoid cropping, we should:

```python
img_translation = cv2.warpAffine(img, translation_matrix,
 (num_cols + 70, num_rows + 110))
```

To move the image to the middle of a bigger image frame


```python
num_rows, num_cols = img.shape[:2]

translation_matrix = np.float32([ [1,0,70], [0,1,110] ])
img_translation = cv2.warpAffine(img, translation_matrix, (num_cols + 70, num_rows + 110))
plt.imshow(img_translation)
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_11_1.png)



```python
translation_matrix = np.float32([ [1,0,-30], [0,1,-50] ])
img_translation = cv2.warpAffine(img_translation, translation_matrix, (num_cols + 70 + 60, num_rows + 110 + 80))
plt.imshow(img_translation)
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_12_1.png)


Moreover, there are two more arguments, borderMode and borderValue, that allow us to __fill up the empty borders of the translation with a pixel extrapolation method__:


```python
img_translation = cv2.warpAffine(img, translation_matrix, (num_cols, num_rows), cv2.INTER_LINEAR, cv2.BORDER_WRAP, 1)
plt.imshow(img_translation)
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_14_1.png)


### Rotation

Using getRotationMatrix2D, we can specify the center point around which the image would be rotated as the first argument, then the angle of rotation in degrees, and a scaling factor for the image at the end. __We use 0.7 to shrink the image by 30% so it fits in the frame.__

![](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/matrix2.png)


```python
rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 30, 0.7)
img_rotation = cv2.warpAffine(img, rotation_matrix, (num_cols, num_rows))
plt.imshow(img_rotation)
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_16_1.png)



```python
num_rows, num_cols = img.shape[:2]

translation_matrix = np.float32([ [1,0,int(0.5*num_cols)], [0,1,int(0.5*num_rows)] ])
rotation_matrix = cv2.getRotationMatrix2D((num_cols, num_rows), 30, 1)

img_translation = cv2.warpAffine(img, translation_matrix, (2*num_cols, 2*num_rows))
img_rotation = cv2.warpAffine(img_translation, rotation_matrix, (num_cols*2, num_rows*2))

plt.imshow(img_rotation)
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_17_1.png)


### Image scaling



```python
img_scaled = cv2.resize(img,None,fx=1.2, fy=1.2, interpolation = cv2.INTER_LINEAR)
plt.imshow(img_scaled)
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_19_1.png)



```python
img_scaled = cv2.resize(img,None,fx=1.2, fy=1.2, interpolation = cv2.INTER_CUBIC)
plt.imshow(img_scaled)
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_20_1.png)



```python
img_scaled = cv2.resize(img, (28, 28), interpolation = cv2.INTER_AREA)
plt.imshow(img_scaled)
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_21_1.png)


### Shear


```python
rows, cols = img.shape[:2]
src_points = np.float32([[0,0], [cols-1,0], [0,rows-1]])
dst_points = np.float32([[0,0], [int(0.6*(cols-1)),0], [int(0.4*(cols-1)),rows-1]])
affine_matrix = cv2.getAffineTransform(src_points, dst_points)
img_output = cv2.warpAffine(img, affine_matrix, (cols,rows))
plt.imshow(img)
plt.imshow(img_output)
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_23_1.png)


### mirror



```python
src_points = np.float32([[0,0], [cols-1,0], [0,rows-1]])
dst_points = np.float32([[cols-1,0], [0,0], [cols-1,rows-1]])
affine_matrix = cv2.getAffineTransform(src_points, dst_points)
img_output = cv2.warpAffine(img, affine_matrix, (cols,rows))
fig=plt.figure()
fig.add_subplot(1, 2, 1)
plt.imshow(img)
fig.add_subplot(1, 2, 2)
plt.imshow(img_output)
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_25_1.png)


### Projective transformations


```python
rows, cols = img.shape[:2]
src_points = np.float32([[0,0], [cols-1,0], [0,rows-1], [cols-1,rows-1]])
dst_points = np.float32([[0,0], [cols-1,0], [int(0.33*cols),rows-1], [int(0.66*cols),rows-1]])
projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
img_output = cv2.warpPerspective(img, projective_matrix, (cols,rows))
plt.imshow(img)
plt.imshow(img_output)
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_27_1.png)



```python
src_points = np.float32([[0,0], [0,rows-1], [cols/2,0],[cols/2,rows-1]])
dst_points = np.float32([[0,100], [0,rows-101], [cols/2,0],[cols/2,rows-1]])
projective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
img_output = cv2.warpPerspective(img, projective_matrix, (cols,rows))
plt.imshow(img)
plt.imshow(img_output)
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_28_1.png)



### Warping


```python
# Vertical wave 
import math
img_output = np.zeros(img.shape, dtype=img.dtype) 
 
for i in range(rows): 
    for j in range(cols): 
        offset_x = int(25.0 * math.sin(2 * 3.14 * i / 180)) 
        offset_y = 0 
        if j+offset_x < rows: 
            img_output[i,j] = img[i,(j+offset_x)%cols] 
        else: 
            img_output[i,j] = 0 
plt.imshow(img) 
plt.imshow(img_output) 
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_30_1.png)



```python
# Horizontal wave 
 
img_output = np.zeros(img.shape, dtype=img.dtype) 
 
for i in range(rows): 
    for j in range(cols): 
        offset_x = 0 
        offset_y = int(16.0 * math.sin(2 * 3.14 * j / 150)) 
        if i+offset_y < rows: 
            img_output[i,j] = img[(i+offset_y)%rows,j] 
        else: 
            img_output[i,j] = 0 
plt.imshow(img_output) 
 
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_31_1.png)



```python
# Both horizontal and vertical  
 
img_output = np.zeros(img.shape, dtype=img.dtype) 
 
for i in range(rows): 
    for j in range(cols): 
        offset_x = int(20.0 * math.sin(2 * 3.14 * i / 150)) 
        offset_y = int(20.0 * math.cos(2 * 3.14 * j / 150)) 
        if i+offset_y < rows and j+offset_x < cols: 
            img_output[i,j] = img[(i+offset_y)%rows,(j+offset_x)%cols] 
        else: 
            img_output[i,j] = 0 
plt.imshow(img_output)  
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_32_1.png)



```python
img_output = np.zeros(img.shape, dtype=img.dtype) 
 
for i in range(rows): 
    for j in range(cols): 
        offset_x = int(128.0 * math.sin(2 * 3.14 * i / (2*cols))) 
        offset_y = 0 
        if j+offset_x < cols: 
            img_output[i,j] = img[i,(j+offset_x)%cols] 
        else: 
            img_output[i,j] = 0 
plt.imshow(img_output) 
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_33_1.png)


### Detecting Edges and Applying Image Filters

### 2D convolution

We basically apply a mathematical operator to each pixel, and change its value in some way. 

To apply this mathematical operator, we use another matrix called a kernel.

The kernel is usually much smaller in size than the input image. 

__For each pixel in the image, we take the kernel and place it on top so that the center of the kernel coincides with the pixel under consideration.__


__We then multiply each value in the kernel matrix with the corresponding values in the image, and then sum it up.__ This is the new value that will be applied to this position in the output image.

Here, __the kernel is called the image filter__ and the process of __applying this kernel to the given image is called image filtering.__


_Depending on the values in the kernel, it performs different functions such as blurring, detecting edges, and so on._


![](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/filter.png)


- identity kernel. 

This kernel doesn't really change the input image

![](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/ik.png)

### Blurring (also called a low pass filter: allow low frequencies, and blocks higher frequencies)

> what does frequency mean in an image?

frequency refers to __the rate of change of pixel values__. 

So we can say that __the sharp edges would be high-frequency content because the pixel values change rapidly in that region.__

thus, plain areas would be low-frequency content. 

So, a low pass filter would try to smooth the edges.

Blurring refers to __averaging the pixel values within a neighborhood.__

We can choose the size of the kernel depending on how much we want to smooth the image,

If we've chosen a bigger size, then you will be averaging over a larger area. This tends to increase the smoothing effect.


- low pass kernel example:

![](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/blurring.png)

- normalization:
    - We are dividing the matrix by 9 because we want the values to sum up to one, because we don't want to artificially increase the intensity value at that pixel's location
    - So, we should normalize the kernel before applying it to an image



```python
rows, cols = img.shape[:2] 

kernel_identity = np.array([[0,0,0], [0,1,0], [0,0,0]]) 
kernel_3x3 = np.ones((3,3), np.float32) / 9.0      # Divide by 9 to normalize the kernel
kernel_5x5 = np.ones((5,5), np.float32) / 25.0     # Divide by 25 to normalize the kernel

fig=plt.figure(figsize=(15,5))

fig.add_subplot(1,3,1)
# value -1 is to maintain source image depth
output = cv2.filter2D(img, -1, kernel_identity) 
plt.imshow(output) 

fig.add_subplot(1,3,2)
output = cv2.filter2D(img, -1, kernel_3x3) 
plt.imshow(output) 

fig.add_subplot(1,3,3)
output = cv2.filter2D(img, -1, kernel_5x5) 
plt.imshow(output) 
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_37_1.png)


#### Size of the kernel versus blurriness

__we can see that the images are keep getting blurrier as we increase the kernel size.__

_The reason for this is because when we increase the kernel size, we are averaging over a larger area. This tends to have a larger blurring effect._



```python
# alternative way
output = cv2.blur(img, (3,3))
plt.imshow(output)
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_39_1.png)


### Motion blur (motion blur kernel averages the pixel values in a particular direction, like a directional low pass filter)

- for example:

![](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/motion.png)

This kernel blurs the image in a horizontal direction. 

The amount of blurring will depend on the size of the kernel. 

So, if we want to make the image blurrier, just pick a bigger size for the kernel. 


```python
size = 15 
 
# generating the kernel 
kernel_motion_blur = np.zeros((size, size)) 
kernel_motion_blur[int((size-1)/2), :] = np.ones(size) 
kernel_motion_blur = kernel_motion_blur / size # normalization

output = cv2.filter2D(img, -1, kernel_motion_blur) 
 
plt.imshow(output) 
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_41_1.png)


### Embossing (饰以浮雕花纹的)

__We basically take each pixel, and replace it with a shadow or a highlight. __




```python
img_emboss_input = cv2.imread('../house.png') 
```


```python
# generating the kernels 
kernel_emboss_1 = np.array([[0,-1,-1], 
                            [1,0,-1], 
                            [1,1,0]]) 
kernel_emboss_2 = np.array([[-1,-1,0], 
                            [-1,0,1], 
                            [0,1,1]]) 
kernel_emboss_3 = np.array([[1,0,0], 
                            [0,0,0], 
                            [0,0,-1]]) 
 
# converting the image to grayscale 
gray_img = cv2.cvtColor(img_emboss_input,cv2.COLOR_BGR2GRAY) 
plt.imshow(gray_img)
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_44_1.png)



```python
# applying the kernels to the grayscale image and adding the offset to produce the shadow
https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_1 = cv2.filter2D(gray_img, -1, kernel_emboss_1) + 128
https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_2 = cv2.filter2D(gray_img, -1, kernel_emboss_2) + 128 
https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_3 = cv2.filter2D(gray_img, -1, kernel_emboss_3) + 128 
```


```python
fig=plt.figure(figsize=(15,5))
fig.add_subplot(1,3,1)
plt.imshow(https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_1) 
fig.add_subplot(1,3,2)
plt.imshow(https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_2) 
fig.add_subplot(1,3,3)
plt.imshow( https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_3) 
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_46_1.png)


### Edge detection

detect sharp edges in the image, and producing a binary image as the output. 

__We can think of edge detection as a high pass filtering operation, which allows high-frequency content to pass through and blocks the low-frequency content.__

Since edges are high-frequency content. In edge detection, we want to retain these edges and discard everything else. 

- Sobel filter

![](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/sobel1.png)


```python
# Attention!! first need to convert normal image to GRAYSCALE!
gray_img = cv2.imread('https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/shapes.png', cv2.IMREAD_GRAYSCALE) 
rows, cols = gray_img.shape 
 
# use depth of cv2.CV_64F.
sobel_horizontal = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=5)

# kernel size can be: 
# 1,3,5 or 7.
sobel_vertical = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=5) 

fig=plt.figure(figsize=(10,5))
fig.add_subplot(1,2,1)
plt.imshow(sobel_horizontal) 
fig.add_subplot(1,2,2)
plt.imshow(sobel_vertical) 
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_48_1.png)


As we can see here, the Sobel filter detects edges in either a horizontal or vertical direction and it doesn't give us a holistic view of all the edges. 

> Solutions?

- Laplacian filter. 



```python
laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
plt.imshow(laplacian)
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_50_1.png)



```python
gray_img = cv2.imread('https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/train.png', cv2.IMREAD_GRAYSCALE) 
laplacian = cv2.Laplacian(gray_img, cv2.CV_64F)
plt.imshow(laplacian)
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_51_1.png)


As we can see in the preceding images, the Laplacian kernel gives rise to a __noisy https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output__

> Solution?

- the Canny edge detector:


```python
canny = cv2.Canny(gray_img, 50, 240) 
plt.imshow(canny)
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_53_1.png)


Canny in opencv takes two numbers as arguments to indicate the thresholds. __The second argument is called the low threshold value, and the third argument is called the high threshold value.__

### Erosion and dilation (primarily defined for binary images, but we can also use them on grayscale images)

__Erosion basically strips out the outermost layer of pixels in a structure, whereas dilation adds an extra layer of pixels to a structure.__


```python
kernel = np.ones((5,5), np.uint8) 
 
img_erosion = cv2.erode(gray_img, kernel, iterations=1) 
img_dilation = cv2.dilate(gray_img, kernel, iterations=1) 
 
fig=plt.figure(figsize=(10,5))
fig.add_subplot(1,2,1)
plt.imshow(img_erosion) 
fig.add_subplot(1,2,2)
plt.imshow(img_dilation) 
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_56_1.png)


### Creating a vignette filter (basically focuses the brightness on a particular part of the image and the other parts look faded)


```python
rows, cols = img.shape[:2] 
 
# generating vignette mask using Gaussian kernels 
kernel_x = cv2.getGaussianKernel(cols,200) # 200: the standard deviation of the Gaussian, and it controls the radius of the bright central region
kernel_y = cv2.getGaussianKernel(rows,200) 
kernel = kernel_y * kernel_x.T 
mask = 255 * kernel / np.linalg.norm(kernel) # if don't scale it up, the image will look black, because all the pixel values will be close to zero after superimposing the mask
output = np.copy(img) 
 
# applying the mask to each channel in the input image 
for i in range(3): 
    output[:,:,i] = output[:,:,i] * mask 
plt.imshow(output) 
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_58_1.png)


### shift focus



```python
kernel_x = cv2.getGaussianKernel(int(1.5*cols),200) 
kernel_y = cv2.getGaussianKernel(int(1.5*rows),200) 
kernel = kernel_y * kernel_x.T 
mask = 255 * kernel / np.linalg.norm(kernel) 
mask = mask[int(0.5*rows):, int(0.5*cols):] 
output = np.copy(img) 
 
# applying the mask to each channel in the input image 
for i in range(3): 
    output[:,:,i] = output[:,:,i] * mask 
plt.imshow(output) 
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_60_1.png)


### Enhancing the contrast in an image

Whenever we capture images in low-light conditions, the images turn out to be dark. 

The reason this happens is because the pixel values tend to concentrate near zero when we capture the images under such conditions. 

- histogram equalization


```python
# equalize the histogram of the input image 
img=cv2.imread("https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/dark.png", 0)
histeq = cv2.equalizeHist(img) 

fig=plt.figure(figsize=(10,5))
fig.add_subplot(1,2,1)
plt.imshow(img) 
fig.add_subplot(1,2,2)
plt.imshow(histeq) 
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_62_1.png)


> For colored image?


```python
img = cv2.imread('../got.png') 
 
img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV) 
 
# equalize the histogram of the Y channel 
img_yuv[0,:,:] = cv2.equalizeHist(img_yuv[0,:,:]) 
 
img_output1 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) 

# equalize the histogram of the Y channel 
img_yuv[:,0,:] = cv2.equalizeHist(img_yuv[:,0,:]) 
 
img_output2 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) 

# equalize the histogram of the Y channel 
img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0]) 
 
img_output3 = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR) 
 
fig=plt.figure(figsize=(16,8))
fig.add_subplot(2,2,1)
plt.imshow(img) 
fig.add_subplot(2,2,2)
plt.imshow(img_output1)
fig.add_subplot(2,2,3)
plt.imshow(img_output2) 
fig.add_subplot(2,2,4)
plt.imshow(img_output3) 
```




    




![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic/output_64_1.png)



### Applying hough transform, to detect the horizon


```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

%matplotlib inline

plt.rcParams["figure.figsize"] = (10,5)

image = mpimg.imread('files/houghLines.jpg')
gray  = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

edges = cv2.Canny(gray, 50,400)

lines = cv2.HoughLines( edges, 1 , np.pi / 180, 200)

rho,theta = lines[7][0]
x  = np.cos(theta)
y  = np.sin(theta)
x0 = x * rho
y0 = y * rho
x1 = int(x0 + 1000 * (-y))
y1 = int(y0 + 1000 * (x))
x2 = int(x0 - 1000 * (-y))
y2 = int(y0 - 1000 * (x))
    
# we now obtained the end points of the lines, and plot them
    
cv2.line(image, (x1, y2), (x2, y2), (0,0,255), 2)

plt.imshow(image)
plt.title('Horizon')
plt.show()
```


![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic2/output_2_0.png)


### Applying hough transform, to circles in an image


```python
image = mpimg.imread('files/houghCircles.jpg')

gray     = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
blurred  = cv2.GaussianBlur(gray, (5,5), 0)
circles  = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 120, param1=50, param2=30, minRadius=50, maxRadius=90)


for i in circles[0,:]:
    #draw the outer circle
    cv2.circle(image, (i[0], i[1]), i[2], (0,255,0), 2)
    
    #draw the center of the circle
    cv2.circle(image, (i[0],i[1]), 2, (0,0,255), 3)
plt.gray()    
plt.imshow(image)
plt.title('Coins')
plt.show()
```


![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic2/output_4_0.png)


### Stretch and Shrink an Image


```python
image = mpimg.imread('files/lena.jpg')

scaleup   = cv2.resize(image, None, fx=2, fy=2,interpolation=cv2.INTER_CUBIC)
scaledown = cv2.resize(image, None, fx=0.5, fy=0.5,interpolation=cv2.INTER_AREA)


plt.subplot(131)
plt.imshow(image)
plt.title("Original Image")


plt.subplot(132)
plt.imshow(scaleup)
plt.title("Stretched Image [2x]")


plt.subplot(133)
plt.imshow(scaledown)
plt.title("Shrinked Image [0.5x]")


plt.tight_layout()

plt.show()
```


![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic2/output_6_0.png)


### Affine and Perspective Transform


```python
affImg  = mpimg.imread("files/affine.jpg")
persImg = mpimg.imread("files/sudoku_small.jpg")

rows1,cols1,ch1 = affImg.shape

apts1 = np.float32([[185,814],[284,493],[882,690]])
apts2 = np.float32([[100,787],[140,455],[786,551]])

M1 = cv2.getAffineTransform(apts1,apts2)
affTrans = cv2.warpAffine(affImg,M1,(cols1,rows1))

ppts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
ppts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M2 = cv2.getPerspectiveTransform(ppts1,ppts2)
persTrans = cv2.warpPerspective(persImg,M2,(300,300))


plt.figure(figsize=(5*2,10))

plt.subplot(221)
plt.imshow(affImg)
plt.title("Original Image")
plt.axis("off")

plt.subplot(222)
plt.imshow(affTrans)
plt.title("Affine Transform")
plt.axis("off")

plt.subplot(223)
plt.imshow(persImg)
plt.title("Original Image")
plt.axis("off")

plt.subplot(224)
plt.imshow(persTrans)
plt.title("Perspective Transform")
plt.axis("off")

plt.tight_layout()

plt.show()
```


![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic2/output_8_0.png)


### Rotation


```python
def rotateImage(image, angle):
    center=tuple(np.array(image.shape[0:2])/2)
    rot_mat = cv2.getRotationMatrix2D(center,angle,1.0)
    return cv2.warpAffine(image, rot_mat, image.shape[0:2],flags=cv2.INTER_LINEAR)

img  = mpimg.imread('files/pikachu.png')
rImg = rotateImage(img,90)

plt.subplot(121)
plt.imshow(img)
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(rImg)
plt.title('Rotated Image')
plt.axis('off')

plt.show()
```


![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic2/output_10_0.png)


### This line suppresses runtime error warnings like divide-by-zero



```python
np.seterr(divide='ignore', invalid='ignore')

image = mpimg.imread('files/checks.jpg')

# Apply sobel filter in X-direction
Gx = cv2.Sobel(image,cv2.CV_64F,1,0,ksize=5)

# Apply sobel filter in Y-direction
Gy = cv2.Sobel(image,cv2.CV_64F,0,1,ksize=5)

# Finding the magnitude of the sobel filter
Gxy = np.sqrt(Gx * Gx + Gy * Gy)

# calculate the angle of the edges, relative to the pixel grid
theta = np.arctan(Gy/Gx)

plt.subplot(231)
plt.imshow(image,cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(232)
plt.imshow(Gx,cmap="gray")
plt.title("GX")
plt.axis("off")

plt.subplot(233)
plt.imshow(Gy,cmap="gray")
plt.title("GY")
plt.axis("off")

plt.subplot(234)
plt.imshow(Gxy,cmap="gray")
plt.title("GXY")
plt.axis("off")

plt.subplot(235)
plt.imshow(theta,cmap="gray")
plt.title("Orientation")
plt.axis("off")

plt.tight_layout()

plt.show()
```


![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic2/output_12_0.png)


### Laplacian Derivative Example


```python
image  = mpimg.imread('files/checks.jpg')
limage = cv2.Laplacian(image,cv2.CV_64F)


plt.subplot(121)
plt.imshow(image,cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(122)
plt.imshow(limage,cmap="gray")
plt.title("Laplacian Derivative")
plt.axis("off")

plt.tight_layout()

plt.show()
```


![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic2/output_14_0.png)


### Histogram Equalization


```python
from PIL import Image
from pylab import *


def histeq(im,nbr_bins=256):
    
    # get image histogram
    imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)

    subplot(222)
    title('Image Histogram')
    plot(imhist)
    
    # getting the cummulative distribution function
    cdf = imhist.cumsum()
    
    cdf = 255*cdf/cdf[-1]  # normalize

    subplot(223)
    title('Cumulative Histogram')
    plot(cdf)    

    # use linear interpolation of cdf to find new pixel values
    im2 = interp(im.flatten(),bins[:-1],cdf)

    return im2.reshape(im.shape),cdf

im = array(Image.open('files/shipwreck.jpg').convert('L'))

subplot(221)
title('Original GrayScale')
imshow(im)

im2,_ = histeq(im)
subplot(224)
title('Histogram Equalized Image')
imshow(im2)

axis('off')

tight_layout()
show()
```


![png](https://raw.githubusercontent.com/karenyyy/data_science/master/opencv/images/basic2/output_16_0.png)

