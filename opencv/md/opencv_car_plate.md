
> What are contours?

- contours are curves joining all the continous points (along the boundary), having same color or intensity;
- useful for shape analysis and obj detection and recogntion
- A contour is a list of points that represent a curve in an image
    - in OpenCV:
        - cv2.findContours()
            - modes:
                - cv2.RETR_EXTERNAL
                - cv2.RETR_LIST
                - cv2.RETR_CCOMP
                - cv2.RETR_TREE
        - cv2.drawContours()


```python
import cv2
import numpy as np
```


```python
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


%matplotlib inline

img = cv2.imread('../images/shapes.png')


image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


print ("There are {0} contours".format(len(contours)))

for cnt in contours:
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    img = cv2.drawContours(img, [box], 0, (0,255,0) ,3)

plt.figure(figsize=(16,8))
plt.imshow(img)
plt.title('Binary Contours in an image')
plt.show()
```

    There are 10 contours



![png](output_2_1.png)


### Appriximating a Contour Shape using Polygon Approximation


```python
img = cv2.imread('../images/shapes.png')
image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
for cnt in contours:
    # eplison - is maximum distance from contour to approximated contour - set as 1% in our case
    # eplison = error_rate * actual_arc_length
    epsilon = 0.01* cv2.arcLength(cnt,True)

    # Use approxPolyDP to approximate a polygon
    approx = cv2.approxPolyDP(cnt, epsilon, True) 
    
    img = cv2.drawContours(img, [approx], 0, (0,255,0) ,3)

plt.figure(figsize=(16,8))
plt.imshow(img)
plt.title('Binary Contours in an image - Polynomial Approximation')
plt.show()
```


![png](output_4_0.png)


### Capturing Hu Moments from Images
- Image moments helps calculate some features like center of mass of the object, area of the object


```python
img = cv2.imread('../images/shapes.png')
image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

for cnt in contours:
    
    M = cv2.moments(cnt)
    
    # finding the center of the contour 
    # m10 / m00 , m01 / m00 -> center
    
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
    center = (cx, cy)
    
    # get the area of the contour using contourArea
    area = cv2.contourArea(cnt)
    
    # getting the contour perimeter using arcLength
    perimeter = cv2.arcLength(cnt, True)
    
    # A -> Area P -> Permimeter
    cv2.putText(img, "A: {0:2.1f}".format(area),center, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.3 ,(255, 0, 0), 3)    
    
    cv2.putText(img, "P: {0:2.1f}".format(perimeter),(cx, cy + 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.3 ,(255, 0, 0), 3) 
    
plt.figure(figsize=(16,8))
plt.imshow(img)
plt.title('Calculating Hu Moments of Shapes in a Binary Image')
plt.show()
```


![png](output_6_0.png)


### Template matching finding a face in an image


```python
img   = mpimg.imread('../images/kchawla.jpg')
image = img.copy()

image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

template = cv2.imread("../images/kc_template.png",0)

w, h = template.shape[::-1]

# Converting the images to uint8 type
image = np.uint8(image)
template = np.uint8(template)

# performing the actual template matching on the image
res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

# calculating the bounds of the template to be plotted in the next step
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# finding the top left and the bottom right corners in the image
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(img,top_left, bottom_right, (0,255,0), 2)

plt.figure("Template Matching", figsize=(20,10))

plt.subplot(121)
plt.gray()
plt.title("Template")
plt.xticks([])
plt.yticks([])
plt.imshow(template)

plt.subplot(122)
plt.title('Template Matching - Find specific face')
plt.xticks([])
plt.yticks([])
plt.imshow(img)

```




    <matplotlib.image.AxesImage at 0x7f2f2c41a240>




![png](output_8_1.png)



```python
img   = mpimg.imread('../images/kchawla.jpg')
image = img.copy()

image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

template = cv2.imread("../images/another.png",0)

w, h = template.shape[::-1]

# Converting the images to uint8 type
image = np.uint8(image)
template = np.uint8(template)

# performing the actual template matching on the image
res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)

# calculating the bounds of the template to be plotted in the next step
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

# finding the top left and the bottom right corners in the image
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)

cv2.rectangle(img,top_left, bottom_right, (0,255,0), 2)

plt.figure("Template Matching", figsize=(20,10))

plt.subplot(121)
plt.gray()
plt.title("Template")
plt.xticks([])
plt.yticks([])
plt.imshow(template)

plt.subplot(122)
plt.title('Template Matching - Find specific face')
plt.xticks([])
plt.yticks([])
plt.imshow(img)

```




    <matplotlib.image.AxesImage at 0x7f2f2c3e6f28>




![png](output_9_1.png)


### Background Subtraction and detecting humans in a CCTV footage


```python
import numpy as np
from IPython.display import clear_output

vid = cv2.VideoCapture("../images/cctv.mp4")

# creating an elliptical kernel, usign the function cv2.getStructuralElement
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

# Initializing the createBackgroundSubtractorMOG2 background substraction instance
fgbg = cv2.createBackgroundSubtractorMOG2()

try:
    while(True):
        ret, frame = vid.read()
        if not ret:
            vid.release()
            break
         
        # Applying the background substraction per frame and also applying the morphological mask for noise removal
        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        
        plt.figure(figsize=(16,8))
        plt.axis('off')
        plt.title("Input Stream")
        plt.imshow(fgmask)
        
        clear_output(wait=True)
except KeyboardInterrupt:
    vid.release()
```


![png](output_11_0.png)



![png](output_11_1.png)



![png](output_11_2.png)



![png](output_11_3.png)



![png](output_11_4.png)



![png](output_11_5.png)



![png](output_11_6.png)



![png](output_11_7.png)



![png](output_11_8.png)



![png](output_11_9.png)



![png](output_11_10.png)



![png](output_11_11.png)



![png](output_11_12.png)



![png](output_11_13.png)



![png](output_11_14.png)



![png](output_11_15.png)



![png](output_11_16.png)



![png](output_11_17.png)



![png](output_11_18.png)



![png](output_11_19.png)



![png](output_11_20.png)



![png](output_11_21.png)



![png](output_11_22.png)



![png](output_11_23.png)



![png](output_11_24.png)



![png](output_11_25.png)



![png](output_11_26.png)



![png](output_11_27.png)



![png](output_11_28.png)



![png](output_11_29.png)



![png](output_11_30.png)



![png](output_11_31.png)



![png](output_11_32.png)



![png](output_11_33.png)



![png](output_11_34.png)



![png](output_11_35.png)



![png](output_11_36.png)



![png](output_11_37.png)



![png](output_11_38.png)



![png](output_11_39.png)



![png](output_11_40.png)



![png](output_11_41.png)



![png](output_11_42.png)



![png](output_11_43.png)



![png](output_11_44.png)



![png](output_11_45.png)



![png](output_11_46.png)



![png](output_11_47.png)



![png](output_11_48.png)



![png](output_11_49.png)



![png](output_11_50.png)



![png](output_11_51.png)



![png](output_11_52.png)



![png](output_11_53.png)



![png](output_11_54.png)



![png](output_11_55.png)



![png](output_11_56.png)



![png](output_11_57.png)



![png](output_11_58.png)



![png](output_11_59.png)



![png](output_11_60.png)



![png](output_11_61.png)



![png](output_11_62.png)



![png](output_11_63.png)



![png](output_11_64.png)



![png](output_11_65.png)



![png](output_11_66.png)



![png](output_11_67.png)



![png](output_11_68.png)



![png](output_11_69.png)



![png](output_11_70.png)



![png](output_11_71.png)



![png](output_11_72.png)



![png](output_11_73.png)



![png](output_11_74.png)



![png](output_11_75.png)



![png](output_11_76.png)



![png](output_11_77.png)



![png](output_11_78.png)



![png](output_11_79.png)



![png](output_11_80.png)



![png](output_11_81.png)



![png](output_11_82.png)



![png](output_11_83.png)



![png](output_11_84.png)



![png](output_11_85.png)



![png](output_11_86.png)



![png](output_11_87.png)



![png](output_11_88.png)



![png](output_11_89.png)



![png](output_11_90.png)



![png](output_11_91.png)



![png](output_11_92.png)



![png](output_11_93.png)



![png](output_11_94.png)



![png](output_11_95.png)



![png](output_11_96.png)



![png](output_11_97.png)



![png](output_11_98.png)



![png](output_11_99.png)



![png](output_11_100.png)



![png](output_11_101.png)



![png](output_11_102.png)



![png](output_11_103.png)



![png](output_11_104.png)



![png](output_11_105.png)



![png](output_11_106.png)



![png](output_11_107.png)



![png](output_11_108.png)



![png](output_11_109.png)



![png](output_11_110.png)



![png](output_11_111.png)



![png](output_11_112.png)



![png](output_11_113.png)



![png](output_11_114.png)



![png](output_11_115.png)



![png](output_11_116.png)



![png](output_11_117.png)



![png](output_11_118.png)



![png](output_11_119.png)



![png](output_11_120.png)



![png](output_11_121.png)



![png](output_11_122.png)



![png](output_11_123.png)



![png](output_11_124.png)



![png](output_11_125.png)



![png](output_11_126.png)



![png](output_11_127.png)



![png](output_11_128.png)



![png](output_11_129.png)



![png](output_11_130.png)



![png](output_11_131.png)



![png](output_11_132.png)



![png](output_11_133.png)



![png](output_11_134.png)



![png](output_11_135.png)



![png](output_11_136.png)



![png](output_11_137.png)



![png](output_11_138.png)



![png](output_11_139.png)



![png](output_11_140.png)



![png](output_11_141.png)



![png](output_11_142.png)



![png](output_11_143.png)



![png](output_11_144.png)



![png](output_11_145.png)



![png](output_11_146.png)



![png](output_11_147.png)



![png](output_11_148.png)



![png](output_11_149.png)



![png](output_11_150.png)



![png](output_11_151.png)



![png](output_11_152.png)



![png](output_11_153.png)



![png](output_11_154.png)



![png](output_11_155.png)



![png](output_11_156.png)



![png](output_11_157.png)



![png](output_11_158.png)



![png](output_11_159.png)



![png](output_11_160.png)



![png](output_11_161.png)



![png](output_11_162.png)



![png](output_11_163.png)



![png](output_11_164.png)



![png](output_11_165.png)



![png](output_11_166.png)



![png](output_11_167.png)



![png](output_11_168.png)



![png](output_11_169.png)



![png](output_11_170.png)



![png](output_11_171.png)



![png](output_11_172.png)



![png](output_11_173.png)



![png](output_11_174.png)



![png](output_11_175.png)



![png](output_11_176.png)



![png](output_11_177.png)



![png](output_11_178.png)



![png](output_11_179.png)



![png](output_11_180.png)



![png](output_11_181.png)



![png](output_11_182.png)



![png](output_11_183.png)



![png](output_11_184.png)



![png](output_11_185.png)



![png](output_11_186.png)



![png](output_11_187.png)



![png](output_11_188.png)



![png](output_11_189.png)



![png](output_11_190.png)



![png](output_11_191.png)



![png](output_11_192.png)



![png](output_11_193.png)



![png](output_11_194.png)



![png](output_11_195.png)



![png](output_11_196.png)



![png](output_11_197.png)



![png](output_11_198.png)



![png](output_11_199.png)



![png](output_11_200.png)



![png](output_11_201.png)



![png](output_11_202.png)



![png](output_11_203.png)



![png](output_11_204.png)



![png](output_11_205.png)



![png](output_11_206.png)



![png](output_11_207.png)



![png](output_11_208.png)



![png](output_11_209.png)



![png](output_11_210.png)



![png](output_11_211.png)



![png](output_11_212.png)



![png](output_11_213.png)



![png](output_11_214.png)



![png](output_11_215.png)



![png](output_11_216.png)



![png](output_11_217.png)



![png](output_11_218.png)



![png](output_11_219.png)



![png](output_11_220.png)



![png](output_11_221.png)



![png](output_11_222.png)



![png](output_11_223.png)



![png](output_11_224.png)



![png](output_11_225.png)



![png](output_11_226.png)



![png](output_11_227.png)



![png](output_11_228.png)



![png](output_11_229.png)



![png](output_11_230.png)



![png](output_11_231.png)



![png](output_11_232.png)



![png](output_11_233.png)



![png](output_11_234.png)



![png](output_11_235.png)



![png](output_11_236.png)



![png](output_11_237.png)



![png](output_11_238.png)



![png](output_11_239.png)



![png](output_11_240.png)



![png](output_11_241.png)



![png](output_11_242.png)



![png](output_11_243.png)



![png](output_11_244.png)



![png](output_11_245.png)



![png](output_11_246.png)



![png](output_11_247.png)



![png](output_11_248.png)



![png](output_11_249.png)



![png](output_11_250.png)



![png](output_11_251.png)



![png](output_11_252.png)



![png](output_11_253.png)



![png](output_11_254.png)



![png](output_11_255.png)



![png](output_11_256.png)



![png](output_11_257.png)



![png](output_11_258.png)



![png](output_11_259.png)



![png](output_11_260.png)



![png](output_11_261.png)



![png](output_11_262.png)



![png](output_11_263.png)



![png](output_11_264.png)



![png](output_11_265.png)



![png](output_11_266.png)



![png](output_11_267.png)



![png](output_11_268.png)



![png](output_11_269.png)



![png](output_11_270.png)



![png](output_11_271.png)



![png](output_11_272.png)



![png](output_11_273.png)



![png](output_11_274.png)



![png](output_11_275.png)



![png](output_11_276.png)



![png](output_11_277.png)



![png](output_11_278.png)



![png](output_11_279.png)



![png](output_11_280.png)



![png](output_11_281.png)



![png](output_11_282.png)



![png](output_11_283.png)



![png](output_11_284.png)



![png](output_11_285.png)



![png](output_11_286.png)



![png](output_11_287.png)



![png](output_11_288.png)



![png](output_11_289.png)



![png](output_11_290.png)



![png](output_11_291.png)



![png](output_11_292.png)



![png](output_11_293.png)



![png](output_11_294.png)



![png](output_11_295.png)



![png](output_11_296.png)



![png](output_11_297.png)



![png](output_11_298.png)



![png](output_11_299.png)



![png](output_11_300.png)



![png](output_11_301.png)



![png](output_11_302.png)



![png](output_11_303.png)



![png](output_11_304.png)



![png](output_11_305.png)



![png](output_11_306.png)



![png](output_11_307.png)



![png](output_11_308.png)



![png](output_11_309.png)



![png](output_11_310.png)



![png](output_11_311.png)



![png](output_11_312.png)



![png](output_11_313.png)



![png](output_11_314.png)



![png](output_11_315.png)



![png](output_11_316.png)



![png](output_11_317.png)



![png](output_11_318.png)



![png](output_11_319.png)



![png](output_11_320.png)



![png](output_11_321.png)



![png](output_11_322.png)



![png](output_11_323.png)



![png](output_11_324.png)



![png](output_11_325.png)



![png](output_11_326.png)



![png](output_11_327.png)



![png](output_11_328.png)



![png](output_11_329.png)



![png](output_11_330.png)



![png](output_11_331.png)



![png](output_11_332.png)



![png](output_11_333.png)



![png](output_11_334.png)



![png](output_11_335.png)



![png](output_11_336.png)



![png](output_11_337.png)



![png](output_11_338.png)



![png](output_11_339.png)



![png](output_11_340.png)



![png](output_11_341.png)



![png](output_11_342.png)



![png](output_11_343.png)



![png](output_11_344.png)



![png](output_11_345.png)



![png](output_11_346.png)



![png](output_11_347.png)



![png](output_11_348.png)



![png](output_11_349.png)



![png](output_11_350.png)



![png](output_11_351.png)



![png](output_11_352.png)



![png](output_11_353.png)



![png](output_11_354.png)



![png](output_11_355.png)



![png](output_11_356.png)



![png](output_11_357.png)



![png](output_11_358.png)



![png](output_11_359.png)



![png](output_11_360.png)



![png](output_11_361.png)



![png](output_11_362.png)



![png](output_11_363.png)



![png](output_11_364.png)



![png](output_11_365.png)



![png](output_11_366.png)



![png](output_11_367.png)



![png](output_11_368.png)



![png](output_11_369.png)



![png](output_11_370.png)



![png](output_11_371.png)



![png](output_11_372.png)



![png](output_11_373.png)



![png](output_11_374.png)



![png](output_11_375.png)



![png](output_11_376.png)



![png](output_11_377.png)



![png](output_11_378.png)



![png](output_11_379.png)



![png](output_11_380.png)



![png](output_11_381.png)



![png](output_11_382.png)



![png](output_11_383.png)



![png](output_11_384.png)



![png](output_11_385.png)



![png](output_11_386.png)



![png](output_11_387.png)



![png](output_11_388.png)



![png](output_11_389.png)



![png](output_11_390.png)



![png](output_11_391.png)



![png](output_11_392.png)



![png](output_11_393.png)



![png](output_11_394.png)



![png](output_11_395.png)



![png](output_11_396.png)



![png](output_11_397.png)



![png](output_11_398.png)



![png](output_11_399.png)



![png](output_11_400.png)



![png](output_11_401.png)



![png](output_11_402.png)



![png](output_11_403.png)



![png](output_11_404.png)



![png](output_11_405.png)



![png](output_11_406.png)



![png](output_11_407.png)



![png](output_11_408.png)



![png](output_11_409.png)



![png](output_11_410.png)



![png](output_11_411.png)



![png](output_11_412.png)



![png](output_11_413.png)



![png](output_11_414.png)



![png](output_11_415.png)



![png](output_11_416.png)



![png](output_11_417.png)



![png](output_11_418.png)



![png](output_11_419.png)



![png](output_11_420.png)



![png](output_11_421.png)



![png](output_11_422.png)



![png](output_11_423.png)



![png](output_11_424.png)



![png](output_11_425.png)



![png](output_11_426.png)



![png](output_11_427.png)



![png](output_11_428.png)



![png](output_11_429.png)



![png](output_11_430.png)



![png](output_11_431.png)



![png](output_11_432.png)



![png](output_11_433.png)



![png](output_11_434.png)



![png](output_11_435.png)



![png](output_11_436.png)



![png](output_11_437.png)



![png](output_11_438.png)



![png](output_11_439.png)



![png](output_11_440.png)



![png](output_11_441.png)



![png](output_11_442.png)



![png](output_11_443.png)



![png](output_11_444.png)



![png](output_11_445.png)



![png](output_11_446.png)



![png](output_11_447.png)



![png](output_11_448.png)



![png](output_11_449.png)



![png](output_11_450.png)



![png](output_11_451.png)



![png](output_11_452.png)



![png](output_11_453.png)



![png](output_11_454.png)



![png](output_11_455.png)



![png](output_11_456.png)



![png](output_11_457.png)



![png](output_11_458.png)



![png](output_11_459.png)



![png](output_11_460.png)



![png](output_11_461.png)



![png](output_11_462.png)



![png](output_11_463.png)



![png](output_11_464.png)



![png](output_11_465.png)



![png](output_11_466.png)



![png](output_11_467.png)



![png](output_11_468.png)



![png](output_11_469.png)



![png](output_11_470.png)



![png](output_11_471.png)



![png](output_11_472.png)



![png](output_11_473.png)



![png](output_11_474.png)



![png](output_11_475.png)



![png](output_11_476.png)



![png](output_11_477.png)



![png](output_11_478.png)



![png](output_11_479.png)



![png](output_11_480.png)



![png](output_11_481.png)



![png](output_11_482.png)



![png](output_11_483.png)



![png](output_11_484.png)



![png](output_11_485.png)



![png](output_11_486.png)



![png](output_11_487.png)



![png](output_11_488.png)



![png](output_11_489.png)



![png](output_11_490.png)



![png](output_11_491.png)



![png](output_11_492.png)



![png](output_11_493.png)



![png](output_11_494.png)



![png](output_11_495.png)



![png](output_11_496.png)



![png](output_11_497.png)



![png](output_11_498.png)



![png](output_11_499.png)



![png](output_11_500.png)



![png](output_11_501.png)



![png](output_11_502.png)



![png](output_11_503.png)



![png](output_11_504.png)



![png](output_11_505.png)



![png](output_11_506.png)



![png](output_11_507.png)



![png](output_11_508.png)



![png](output_11_509.png)



![png](output_11_510.png)



![png](output_11_511.png)



![png](output_11_512.png)



![png](output_11_513.png)



![png](output_11_514.png)



![png](output_11_515.png)



![png](output_11_516.png)



![png](output_11_517.png)



![png](output_11_518.png)



![png](output_11_519.png)



![png](output_11_520.png)



![png](output_11_521.png)



![png](output_11_522.png)



![png](output_11_523.png)



![png](output_11_524.png)



![png](output_11_525.png)



![png](output_11_526.png)



![png](output_11_527.png)



![png](output_11_528.png)



![png](output_11_529.png)



![png](output_11_530.png)



![png](output_11_531.png)



![png](output_11_532.png)



![png](output_11_533.png)



![png](output_11_534.png)



![png](output_11_535.png)



![png](output_11_536.png)



![png](output_11_537.png)



![png](output_11_538.png)



![png](output_11_539.png)



![png](output_11_540.png)



![png](output_11_541.png)



![png](output_11_542.png)



![png](output_11_543.png)



![png](output_11_544.png)



![png](output_11_545.png)



![png](output_11_546.png)



![png](output_11_547.png)



![png](output_11_548.png)



![png](output_11_549.png)



![png](output_11_550.png)



![png](output_11_551.png)



![png](output_11_552.png)



![png](output_11_553.png)



![png](output_11_554.png)



![png](output_11_555.png)



![png](output_11_556.png)



![png](output_11_557.png)



![png](output_11_558.png)



![png](output_11_559.png)



![png](output_11_560.png)



![png](output_11_561.png)



![png](output_11_562.png)



![png](output_11_563.png)



![png](output_11_564.png)



![png](output_11_565.png)



![png](output_11_566.png)



![png](output_11_567.png)



![png](output_11_568.png)



![png](output_11_569.png)



![png](output_11_570.png)



![png](output_11_571.png)



![png](output_11_572.png)



![png](output_11_573.png)



![png](output_11_574.png)



![png](output_11_575.png)



![png](output_11_576.png)



![png](output_11_577.png)



![png](output_11_578.png)



![png](output_11_579.png)



![png](output_11_580.png)



![png](output_11_581.png)



![png](output_11_582.png)



![png](output_11_583.png)



![png](output_11_584.png)



![png](output_11_585.png)



![png](output_11_586.png)



![png](output_11_587.png)



![png](output_11_588.png)



![png](output_11_589.png)



![png](output_11_590.png)



![png](output_11_591.png)



![png](output_11_592.png)



![png](output_11_593.png)



![png](output_11_594.png)



![png](output_11_595.png)



![png](output_11_596.png)



![png](output_11_597.png)



![png](output_11_598.png)



![png](output_11_599.png)



![png](output_11_600.png)



![png](output_11_601.png)



![png](output_11_602.png)



![png](output_11_603.png)



![png](output_11_604.png)



![png](output_11_605.png)



![png](output_11_606.png)



![png](output_11_607.png)



![png](output_11_608.png)



![png](output_11_609.png)



![png](output_11_610.png)



![png](output_11_611.png)



![png](output_11_612.png)



![png](output_11_613.png)



![png](output_11_614.png)



![png](output_11_615.png)



![png](output_11_616.png)



![png](output_11_617.png)



![png](output_11_618.png)



![png](output_11_619.png)



![png](output_11_620.png)



![png](output_11_621.png)



![png](output_11_622.png)



![png](output_11_623.png)



![png](output_11_624.png)



![png](output_11_625.png)



![png](output_11_626.png)



![png](output_11_627.png)



![png](output_11_628.png)



![png](output_11_629.png)



![png](output_11_630.png)



![png](output_11_631.png)



![png](output_11_632.png)



![png](output_11_633.png)



![png](output_11_634.png)



![png](output_11_635.png)



![png](output_11_636.png)



![png](output_11_637.png)



![png](output_11_638.png)



![png](output_11_639.png)



![png](output_11_640.png)



![png](output_11_641.png)



![png](output_11_642.png)



![png](output_11_643.png)



![png](output_11_644.png)



![png](output_11_645.png)



![png](output_11_646.png)



![png](output_11_647.png)



![png](output_11_648.png)



![png](output_11_649.png)



![png](output_11_650.png)



![png](output_11_651.png)



![png](output_11_652.png)



![png](output_11_653.png)



![png](output_11_654.png)



![png](output_11_655.png)



![png](output_11_656.png)



![png](output_11_657.png)



![png](output_11_658.png)



![png](output_11_659.png)



![png](output_11_660.png)



![png](output_11_661.png)



![png](output_11_662.png)



![png](output_11_663.png)



![png](output_11_664.png)



![png](output_11_665.png)



![png](output_11_666.png)



![png](output_11_667.png)



![png](output_11_668.png)



![png](output_11_669.png)



![png](output_11_670.png)



![png](output_11_671.png)



![png](output_11_672.png)



![png](output_11_673.png)



![png](output_11_674.png)



![png](output_11_675.png)



![png](output_11_676.png)



![png](output_11_677.png)



![png](output_11_678.png)



![png](output_11_679.png)



![png](output_11_680.png)



![png](output_11_681.png)



![png](output_11_682.png)



![png](output_11_683.png)



![png](output_11_684.png)



![png](output_11_685.png)



![png](output_11_686.png)



![png](output_11_687.png)



![png](output_11_688.png)



![png](output_11_689.png)



![png](output_11_690.png)



![png](output_11_691.png)



![png](output_11_692.png)



![png](output_11_693.png)



![png](output_11_694.png)



![png](output_11_695.png)



![png](output_11_696.png)



![png](output_11_697.png)



![png](output_11_698.png)



![png](output_11_699.png)



![png](output_11_700.png)



![png](output_11_701.png)



![png](output_11_702.png)



![png](output_11_703.png)



![png](output_11_704.png)



![png](output_11_705.png)



![png](output_11_706.png)



![png](output_11_707.png)



![png](output_11_708.png)



![png](output_11_709.png)



![png](output_11_710.png)



![png](output_11_711.png)



![png](output_11_712.png)



![png](output_11_713.png)



![png](output_11_714.png)



![png](output_11_715.png)



![png](output_11_716.png)



![png](output_11_717.png)



![png](output_11_718.png)



![png](output_11_719.png)



![png](output_11_720.png)



![png](output_11_721.png)



![png](output_11_722.png)



![png](output_11_723.png)



![png](output_11_724.png)



![png](output_11_725.png)



![png](output_11_726.png)



![png](output_11_727.png)



![png](output_11_728.png)



![png](output_11_729.png)



![png](output_11_730.png)



![png](output_11_731.png)



![png](output_11_732.png)



![png](output_11_733.png)



![png](output_11_734.png)



![png](output_11_735.png)



![png](output_11_736.png)



![png](output_11_737.png)



![png](output_11_738.png)



![png](output_11_739.png)



![png](output_11_740.png)



![png](output_11_741.png)



![png](output_11_742.png)



![png](output_11_743.png)



![png](output_11_744.png)



![png](output_11_745.png)



![png](output_11_746.png)



![png](output_11_747.png)



![png](output_11_748.png)



![png](output_11_749.png)



![png](output_11_750.png)



![png](output_11_751.png)



![png](output_11_752.png)



![png](output_11_753.png)



![png](output_11_754.png)



![png](output_11_755.png)



![png](output_11_756.png)



![png](output_11_757.png)



![png](output_11_758.png)



![png](output_11_759.png)



![png](output_11_760.png)



![png](output_11_761.png)



![png](output_11_762.png)



![png](output_11_763.png)



![png](output_11_764.png)



![png](output_11_765.png)



![png](output_11_766.png)



![png](output_11_767.png)



![png](output_11_768.png)



![png](output_11_769.png)



![png](output_11_770.png)



![png](output_11_771.png)



![png](output_11_772.png)



![png](output_11_773.png)



![png](output_11_774.png)



![png](output_11_775.png)



![png](output_11_776.png)



![png](output_11_777.png)



![png](output_11_778.png)



![png](output_11_779.png)



![png](output_11_780.png)



![png](output_11_781.png)



![png](output_11_782.png)



![png](output_11_783.png)



![png](output_11_784.png)



![png](output_11_785.png)



![png](output_11_786.png)



![png](output_11_787.png)



![png](output_11_788.png)



![png](output_11_789.png)



![png](output_11_790.png)



![png](output_11_791.png)



![png](output_11_792.png)



![png](output_11_793.png)



![png](output_11_794.png)



![png](output_11_795.png)



![png](output_11_796.png)



![png](output_11_797.png)



![png](output_11_798.png)



![png](output_11_799.png)



![png](output_11_800.png)



![png](output_11_801.png)



![png](output_11_802.png)



![png](output_11_803.png)



![png](output_11_804.png)



![png](output_11_805.png)



![png](output_11_806.png)



![png](output_11_807.png)



![png](output_11_808.png)



![png](output_11_809.png)



![png](output_11_810.png)



![png](output_11_811.png)



![png](output_11_812.png)



![png](output_11_813.png)



![png](output_11_814.png)



![png](output_11_815.png)



![png](output_11_816.png)



![png](output_11_817.png)



![png](output_11_818.png)



![png](output_11_819.png)



![png](output_11_820.png)



![png](output_11_821.png)



![png](output_11_822.png)



![png](output_11_823.png)



![png](output_11_824.png)



![png](output_11_825.png)



![png](output_11_826.png)



![png](output_11_827.png)



![png](output_11_828.png)



![png](output_11_829.png)



![png](output_11_830.png)



![png](output_11_831.png)



![png](output_11_832.png)



![png](output_11_833.png)



![png](output_11_834.png)



![png](output_11_835.png)



![png](output_11_836.png)



![png](output_11_837.png)



![png](output_11_838.png)



![png](output_11_839.png)



![png](output_11_840.png)



![png](output_11_841.png)



![png](output_11_842.png)



![png](output_11_843.png)



![png](output_11_844.png)



![png](output_11_845.png)



![png](output_11_846.png)



![png](output_11_847.png)



![png](output_11_848.png)



![png](output_11_849.png)



![png](output_11_850.png)



![png](output_11_851.png)



![png](output_11_852.png)



![png](output_11_853.png)



![png](output_11_854.png)



![png](output_11_855.png)



![png](output_11_856.png)



![png](output_11_857.png)



![png](output_11_858.png)



![png](output_11_859.png)



![png](output_11_860.png)



![png](output_11_861.png)



![png](output_11_862.png)



![png](output_11_863.png)



![png](output_11_864.png)



![png](output_11_865.png)



![png](output_11_866.png)



![png](output_11_867.png)



![png](output_11_868.png)



![png](output_11_869.png)



![png](output_11_870.png)



![png](output_11_871.png)



![png](output_11_872.png)



![png](output_11_873.png)



![png](output_11_874.png)



![png](output_11_875.png)



![png](output_11_876.png)



![png](output_11_877.png)



![png](output_11_878.png)



![png](output_11_879.png)



![png](output_11_880.png)



![png](output_11_881.png)



![png](output_11_882.png)



![png](output_11_883.png)



![png](output_11_884.png)



![png](output_11_885.png)



![png](output_11_886.png)



![png](output_11_887.png)



![png](output_11_888.png)



![png](output_11_889.png)



![png](output_11_890.png)



![png](output_11_891.png)



![png](output_11_892.png)



![png](output_11_893.png)



![png](output_11_894.png)



![png](output_11_895.png)



![png](output_11_896.png)



![png](output_11_897.png)



![png](output_11_898.png)



![png](output_11_899.png)



![png](output_11_900.png)



![png](output_11_901.png)



![png](output_11_902.png)



![png](output_11_903.png)



![png](output_11_904.png)



![png](output_11_905.png)



![png](output_11_906.png)



![png](output_11_907.png)



![png](output_11_908.png)



![png](output_11_909.png)



![png](output_11_910.png)



![png](output_11_911.png)



![png](output_11_912.png)



![png](output_11_913.png)



![png](output_11_914.png)



![png](output_11_915.png)



![png](output_11_916.png)



![png](output_11_917.png)



![png](output_11_918.png)



![png](output_11_919.png)



![png](output_11_920.png)



![png](output_11_921.png)



![png](output_11_922.png)



![png](output_11_923.png)



![png](output_11_924.png)



![png](output_11_925.png)



![png](output_11_926.png)



![png](output_11_927.png)



![png](output_11_928.png)



![png](output_11_929.png)



![png](output_11_930.png)



![png](output_11_931.png)



![png](output_11_932.png)



![png](output_11_933.png)



![png](output_11_934.png)



![png](output_11_935.png)



![png](output_11_936.png)



![png](output_11_937.png)



![png](output_11_938.png)



![png](output_11_939.png)



![png](output_11_940.png)



![png](output_11_941.png)



![png](output_11_942.png)



![png](output_11_943.png)



![png](output_11_944.png)



![png](output_11_945.png)



![png](output_11_946.png)



![png](output_11_947.png)



![png](output_11_948.png)



![png](output_11_949.png)



![png](output_11_950.png)



![png](output_11_951.png)



![png](output_11_952.png)



![png](output_11_953.png)



![png](output_11_954.png)



![png](output_11_955.png)



![png](output_11_956.png)



![png](output_11_957.png)



![png](output_11_958.png)



![png](output_11_959.png)



![png](output_11_960.png)



![png](output_11_961.png)



![png](output_11_962.png)



![png](output_11_963.png)



![png](output_11_964.png)



![png](output_11_965.png)



![png](output_11_966.png)



![png](output_11_967.png)



![png](output_11_968.png)



![png](output_11_969.png)



![png](output_11_970.png)



![png](output_11_971.png)



![png](output_11_972.png)



![png](output_11_973.png)



![png](output_11_974.png)



![png](output_11_975.png)



![png](output_11_976.png)



![png](output_11_977.png)



![png](output_11_978.png)



![png](output_11_979.png)



![png](output_11_980.png)



![png](output_11_981.png)



![png](output_11_982.png)



![png](output_11_983.png)



![png](output_11_984.png)



![png](output_11_985.png)



![png](output_11_986.png)



![png](output_11_987.png)



![png](output_11_988.png)



![png](output_11_989.png)



![png](output_11_990.png)



![png](output_11_991.png)



![png](output_11_992.png)



![png](output_11_993.png)



```python
img = mpimg.imread('../images/coins.jpg')


image = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


image = cv2.GaussianBlur(image, (19, 19), 0)


_,image = cv2.threshold(image,225,255,cv2.THRESH_BINARY_INV)


kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=2)
     

dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)


ret, image = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
image = np.uint8(image)


_, contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


for i,cnt in enumerate(contours):
    
    (x,y),_ = cv2.minEnclosingCircle(cnt)
    center = (int(x),int(y))
    cv2.putText(img, "Coin {0}".format(i+1),center, cv2.FONT_HERSHEY_COMPLEX_SMALL, 2 ,(0, 255, 0), 3);

plt.imshow(img)
plt.title('Coin Count Demo')
plt.show()
```


![png](output_12_0.png)


### Mean Shift Segmentation of an Image

- mean-shift is a non-parametric feature-space analysis technique to partition image into semantically meaningful regions done by clustering pixels in an image
- apply mean-shift procedure to the plotted points after mapping an image on a feature space
- continue the process till each point shifts towards a high density region until convergence


```python
import math
from scipy import ndimage
from random import randint
import matplotlib.pyplot as plt
img = mpimg.imread('../images/butterfly.jpg')

# Mode = 1 indicates that thresholding should be done based on H
# Mode = 2 indicates that thresholding should be done based on Hs and Hr
Mode = 2

# Set appropriate values for H,Hs,Hr and Iter
H = 90
Hr = 90
Hs = 90
Iter = 100

opImg = np.zeros(img.shape,np.uint8)
boundaryImg = np.zeros(img.shape,np.uint8)
```


```python
# param--row : Row of Feature matrix (a pixel)
# param--matrix : Feature matrix
# param--mode : mode=1 uses 1 value of threshold that is H
#               mode=2 uses 2 values of thresholds
#                      Hr for threshold in range domain
#                      Hs for threshold in spatial domain
# returns--neighbors : List of indices of F which are neighbors to the row
def getNeighbors(row,matrix,mode=1):
    neighbors = []
    for i in range(0,len(matrix)):
        cPixel = matrix[i]
        # if mode is 1, we threshold using H
        if (mode == 1):
            d = math.sqrt(sum((cPixel-row)**2))
            if(d<H):
                 neighbors.append(i)
        # otherwise, we threshold using Hr and Hs
        else:
            r = math.sqrt(sum((cPixel[:3]-row[:3])**2))
            s = math.sqrt(sum((cPixel[3:5]-row[3:5])**2))
            if(s < Hs and r < Hr ):
                neighbors.append(i)
    return neighbors
```


```python
# Method markPixels
# Deletes the pixel from the Feature matrix
# Marks the pixel in the output image with the mean intensity
# param--neighbors : Indices of pixels (from F) to be marked
# param--mean : Range and spatial properties for the pixel to be marked
# param--matrix : Feature matrix
# param--cluster : Cluster number
def markPixels(neighbors,mean,matrix,cluster):
    for i in neighbors:
        cPixel = matrix[i]
        x=cPixel[3]
        y=cPixel[4]
        opImg[x][y] = np.array(mean[:3],np.uint8)
        boundaryImg[x][y] = cluster
    return np.delete(matrix,neighbors,axis=0)
```


```python
# Method calculateMean
# Calculates mean of all the neighbors and returns a 
# mean vector
# param--neighbors : List of indices of pixels (from F)
# param--matrix : Feature matrix
# returns--mean : Vector of mean of spatial and range properties
def calculateMean(neighbors,matrix):
    neighbors = matrix[neighbors]
    r=neighbors[:,:1]
    g=neighbors[:,1:2]
    b=neighbors[:,2:3]
    x=neighbors[:,3:4]
    y=neighbors[:,4:5]
    mean = np.array([np.mean(r),np.mean(g),np.mean(b),np.mean(x),np.mean(y)])
    return mean
```


```python
# Method createFeatureMatrix
# Creates a Feature matrix of the image 
# as list of [r,g,b,x,y] for each pixel
# param--img : Image for which we wish to comute Feature matrix
# return--F : Feature matrix
def createFeatureMatrix(img):
    h,w,d = img.shape
    F = []
    for row in range(0,h):
        for col in range(0,w):
            r,g,b = img[row][col]
            F.append([r,g,b,row,col])
    F = np.array(F)
    return F
```


```python
def performMeanShift(img):
    
    clusters = 0
   
    F = createFeatureMatrix(img)
    
    # Iterate over our Feature matrix until it is exhausted
    while(len(F) > 0):
        
        # choose a random row
        randomIndex = randint(0, len(F) - 1)
        
        row = F[randomIndex]
        
        # Cache the row as our initial mean
        initialMean = row
        
        # Group all the neighbors based on the threshold H
        # H can be a single value or two values or range and
        # spatial fomain
        neighbors = getNeighbors(row, F, Mode)
        
        # If we get only 1 neighbor, which is the pixel itself,
        # We can directly mark it in our output image without calculating the shift
        print(len(neighbors))
        
        if(len(neighbors) == 1):
            F = markPixels(neighbors, initialMean, F, clusters)
            clusters+=1
            continue
        
        
        # Calculating the mean and performing the mean shift
    
        mean = calculateMean(neighbors, F)
        
        # Calculate mean shift based on the initial mean
        
        meanShift = abs(mean - initialMean)
            
        # If the mean is below an acceptable value (Iter), we have found the cluseter else we re-iterate
        if(np.mean(meanShift)<Iter):
            F = markPixels(neighbors,mean,F,clusters)
            clusters+=1
            
    return clusters

clusters = performMeanShift(img)
origlabelledImage, orignumobjects = ndimage.label(opImg)
     
plt.figure("Mean Shift Segmentation",figsize=(10,10))
    
    
plt.subplot(211)
plt.imshow(img)
plt.xticks([]), plt.yticks([])
    
plt.subplot(212)
plt.imshow(opImg)
plt.xticks([]), plt.yticks([])
    
    
plt.show()
```

    16666
    6925
    11110
    14615
    18398
    2798
    11750
    4814
    10027
    8041
    6506
    10372
    743
    11084
    2117
    6155
    1285
    1528
    2782
    339
    1164
    586
    587
    1662
    830
    1615
    970
    281
    1015
    3773
    372
    540
    420
    100
    196
    297
    217
    469
    608
    153
    298
    82
    120
    300
    305
    143
    81
    39
    45
    214
    107
    133
    10
    29
    46
    46
    42
    192
    84
    43
    45
    62
    7
    32
    34
    10
    5
    17
    7
    6
    2
    4



![png](output_19_1.png)



```python
# Mean Shift Example for object tracking
import numpy as np
import cv2
from IPython.display import clear_output

%pylab inline 

vid = cv2.VideoCapture('../images/objTrack.mp4')

# take first frame of the video
ret,frame = vid.read()
frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

# setup initial location of window
r,h,c,w = 210,180,470,50  # simply hardcoded the values
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r: r+h, c: c+w]

hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# we create a histogram of the ROI and then normalize it
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0,180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

try:
    while(True):
        # Capture frame-by-frame
        ret, frame = vid.read()
    
        if not ret:
            vid.release()
            break
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # we perform the backprojection between the current frame and the histogram of ROI
        # and apply meanshift to get the new location
        dst = cv2.calcBackProject([hsv], [0], roi_hist, [0,180],1)
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        
        # Drawing the tracking window on the image
        x,y,w,h = track_window
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img2  = cv2.rectangle(frame, (x,y) , (x+w, y+h), 255, 2)
        
        axis('off')
        title("Input Stream")
        imshow(img2)
        show()
        
        # Display the frame until new frame is available
        clear_output(wait=True)
        
except KeyboardInterrupt:
    vid.release()
```


![png](output_20_0.png)

