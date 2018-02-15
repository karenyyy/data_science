

```python
from urllib.request import urlopen
from urllib.error import HTTPError
from bs4 import BeautifulSoup
import sys


def getTitle(url):
    try:
        html = urlopen(url)
    except HTTPError as e:
        print(e)
        return None
    try:
        bsObj = BeautifulSoup(html, "html.parser")
        title = bsObj.body.h1
    except AttributeError as e:
        return None
    return title.text
```


```python
getTitle("https://karenyyyme.herokuapp.com/")
```




    "Karenyyy's Personal Site"




```python
s={""}
for i in range(1,7):
    html = urlopen("https://karenyyyme.herokuapp.com/?page={}".format(i))
    bsObj = BeautifulSoup(html, "html.parser")
    nameList = bsObj.findAll("span", {"class":"post-category"})
    for name in nameList:
        s = s | {name.text}
[i.strip() for i in s if i !='']
```




    ['git',
     'web development',
     'spark_scala',
     'nvidia driver',
     'pytorch',
     'basic machine learning',
     'CV']



try scraping other's code:


```python
html = urlopen("http://blog.csdn.net/ZengDong_1991/article/details/51491606")
bsObj = BeautifulSoup(html, "html.parser")
all_code = bsObj.findAll("pre", {"class": "prettyprint"})
for i in range(len(all_code)):
    print(all_code[i].get_text())
```

    #include <stdio.h>
    #include <cv.h>
    #include <highgui.h>
    
    CvHaarClassifierCascade *cascade;
    CvMemStorage            *storage;
    
    void detect(IplImage *img);
    
    int main(int argc, char** argv)
    {
        CvCapture *capture;
        IplImage  *frame;
        int input_resize_percent = 100;
    
        cascade = (CvHaarClassifierCascade*)cvLoad("cars3.xml", 0, 0, 0);
        storage = cvCreateMemStorage(0);
        capture = cvCaptureFromAVI("video1.avi");
    
        assert(cascade && storage && capture);
    
        cvNamedWindow("video", 1);
    
        const int KEY_SPACE = 32;
        const int KEY_ESC = 27;
    
        int key = 0;
        do
        {
            frame = cvQueryFrame(capture);
    
            if (!frame)
                break;
    
            //   cvResize(frame1, frame);
    
            detect(frame);
    
            key = cvWaitKey(10);
    
            if (key == KEY_SPACE)
                key = cvWaitKey(0);
    
            if (key == KEY_ESC)
                break;
    
        } while (1);
    
        cvDestroyAllWindows();
        cvReleaseImage(&frame);
        cvReleaseCapture(&capture);
        cvReleaseHaarClassifierCascade(&cascade);
        cvReleaseMemStorage(&storage);
    
        return 0;
    }
    
    void detect(IplImage *img)
    {
        CvSize img_size = cvGetSize(img);
        CvSeq *object = cvHaarDetectObjects(
            img,
            cascade,
            storage,
            1.1, //1.1,//1.5, //-------------------SCALE FACTOR
            1, //2        //------------------MIN NEIGHBOURS
            0, //CV_HAAR_DO_CANNY_PRUNING
            cvSize(0, 0),//cvSize( 30,30), // ------MINSIZE
            img_size //cvSize(70,70)//cvSize(640,480)  //---------MAXSIZE
            );
    
        std::cout << "Total: " << object->total << " cars" << std::endl;
        for (int i = 0; i < (object ? object->total : 0); i++)
        {
            CvRect *r = (CvRect*)cvGetSeqElem(object, i);
            cvRectangle(img,
                cvPoint(r->x, r->y),
                cvPoint(r->x + r->width, r->y + r->height),
                CV_RGB(255, 0, 0), 2, 8, 0);
        }
    
        cvShowImage("video", img);
    }


find children:
    - example: find all tables in one post:


```python
html = urlopen("https://karenyyyme.herokuapp.com/post/76/")
bsObj = BeautifulSoup(html, "html.parser")

for i in bsObj.find_all("table",{"class":"dataframe"}):
    for j in i.children:
        print(j)
```

    
    
    <thead>
    <tr style="text-align: right;">
    <th></th>
    <th>method</th>
    <th>number</th>
    <th>orbital_period</th>
    <th>mass</th>
    <th>distance</th>
    <th>year</th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th>0</th>
    <td>Radial Velocity</td>
    <td>1</td>
    <td>269.300</td>
    <td>7.10</td>
    <td>77.40</td>
    <td>2006</td>
    </tr>
    <tr>
    <th>1</th>
    <td>Radial Velocity</td>
    <td>1</td>
    <td>874.774</td>
    <td>2.21</td>
    <td>56.95</td>
    <td>2008</td>
    </tr>
    <tr>
    <th>2</th>
    <td>Radial Velocity</td>
    <td>1</td>
    <td>763.000</td>
    <td>2.60</td>
    <td>19.84</td>
    <td>2011</td>
    </tr>
    <tr>
    <th>3</th>
    <td>Radial Velocity</td>
    <td>1</td>
    <td>326.030</td>
    <td>19.40</td>
    <td>110.62</td>
    <td>2007</td>
    </tr>
    <tr>
    <th>4</th>
    <td>Radial Velocity</td>
    <td>1</td>
    <td>516.220</td>
    <td>10.50</td>
    <td>119.47</td>
    <td>2009</td>
    </tr>
    </tbody>
    
    
    
    
    <thead>
    <tr style="text-align: right;">
    <th></th>
    <th>number</th>
    <th>orbital_period</th>
    <th>mass</th>
    <th>distance</th>
    <th>year</th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th>count</th>
    <td>498.00000</td>
    <td>498.000000</td>
    <td>498.000000</td>
    <td>498.000000</td>
    <td>498.000000</td>
    </tr>
    <tr>
    <th>mean</th>
    <td>1.73494</td>
    <td>835.778671</td>
    <td>2.509320</td>
    <td>52.068213</td>
    <td>2007.377510</td>
    </tr>
    <tr>
    <th>std</th>
    <td>1.17572</td>
    <td>1469.128259</td>
    <td>3.636274</td>
    <td>46.596041</td>
    <td>4.167284</td>
    </tr>
    <tr>
    <th>min</th>
    <td>1.00000</td>
    <td>1.328300</td>
    <td>0.003600</td>
    <td>1.350000</td>
    <td>1989.000000</td>
    </tr>
    <tr>
    <th>25%</th>
    <td>1.00000</td>
    <td>38.272250</td>
    <td>0.212500</td>
    <td>24.497500</td>
    <td>2005.000000</td>
    </tr>
    <tr>
    <th>50%</th>
    <td>1.00000</td>
    <td>357.000000</td>
    <td>1.245000</td>
    <td>39.940000</td>
    <td>2009.000000</td>
    </tr>
    <tr>
    <th>75%</th>
    <td>2.00000</td>
    <td>999.600000</td>
    <td>2.867500</td>
    <td>59.332500</td>
    <td>2011.000000</td>
    </tr>
    <tr>
    <th>max</th>
    <td>6.00000</td>
    <td>17337.500000</td>
    <td>25.000000</td>
    <td>354.000000</td>
    <td>2014.000000</td>
    </tr>
    </tbody>
    
    
    
    
    <thead>
    <tr style="text-align: right;">
    <th></th>
    <th>count</th>
    <th>mean</th>
    <th>std</th>
    <th>min</th>
    <th>25%</th>
    <th>50%</th>
    <th>75%</th>
    <th>max</th>
    </tr>
    <tr>
    <th>method</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th>Astrometry</th>
    <td>2</td>
    <td>2011.500000</td>
    <td>2.121320</td>
    <td>2010</td>
    <td>2010.75</td>
    <td>2011.5</td>
    <td>2012.25</td>
    <td>2013</td>
    </tr>
    <tr>
    <th>Eclipse Timing Variations</th>
    <td>9</td>
    <td>2010.000000</td>
    <td>1.414214</td>
    <td>2008</td>
    <td>2009.00</td>
    <td>2010.0</td>
    <td>2011.00</td>
    <td>2012</td>
    </tr>
    <tr>
    <th>Imaging</th>
    <td>38</td>
    <td>2009.131579</td>
    <td>2.781901</td>
    <td>2004</td>
    <td>2008.00</td>
    <td>2009.0</td>
    <td>2011.00</td>
    <td>2013</td>
    </tr>
    <tr>
    <th>Microlensing</th>
    <td>23</td>
    <td>2009.782609</td>
    <td>2.859697</td>
    <td>2004</td>
    <td>2008.00</td>
    <td>2010.0</td>
    <td>2012.00</td>
    <td>2013</td>
    </tr>
    <tr>
    <th>Orbital Brightness Modulation</th>
    <td>3</td>
    <td>2011.666667</td>
    <td>1.154701</td>
    <td>2011</td>
    <td>2011.00</td>
    <td>2011.0</td>
    <td>2012.00</td>
    <td>2013</td>
    </tr>
    <tr>
    <th>Pulsar Timing</th>
    <td>5</td>
    <td>1998.400000</td>
    <td>8.384510</td>
    <td>1992</td>
    <td>1992.00</td>
    <td>1994.0</td>
    <td>2003.00</td>
    <td>2011</td>
    </tr>
    <tr>
    <th>Pulsation Timing Variations</th>
    <td>1</td>
    <td>2007.000000</td>
    <td>NaN</td>
    <td>2007</td>
    <td>2007.00</td>
    <td>2007.0</td>
    <td>2007.00</td>
    <td>2007</td>
    </tr>
    <tr>
    <th>Radial Velocity</th>
    <td>553</td>
    <td>2007.518987</td>
    <td>4.249052</td>
    <td>1989</td>
    <td>2005.00</td>
    <td>2009.0</td>
    <td>2011.00</td>
    <td>2014</td>
    </tr>
    <tr>
    <th>Transit</th>
    <td>397</td>
    <td>2011.236776</td>
    <td>2.077867</td>
    <td>2002</td>
    <td>2010.00</td>
    <td>2012.0</td>
    <td>2013.00</td>
    <td>2014</td>
    </tr>
    <tr>
    <th>Transit Timing Variations</th>
    <td>4</td>
    <td>2012.500000</td>
    <td>1.290994</td>
    <td>2011</td>
    <td>2011.75</td>
    <td>2012.5</td>
    <td>2013.25</td>
    <td>2014</td>
    </tr>
    </tbody>
    
    
    
    
    <thead>
    <tr style="text-align: right;">
    <th></th>
    <th>key</th>
    <th>data1</th>
    <th>data2</th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th>0</th>
    <td>A</td>
    <td>0</td>
    <td>5</td>
    </tr>
    <tr>
    <th>1</th>
    <td>B</td>
    <td>1</td>
    <td>0</td>
    </tr>
    <tr>
    <th>2</th>
    <td>C</td>
    <td>2</td>
    <td>3</td>
    </tr>
    <tr>
    <th>3</th>
    <td>A</td>
    <td>3</td>
    <td>3</td>
    </tr>
    <tr>
    <th>4</th>
    <td>B</td>
    <td>4</td>
    <td>7</td>
    </tr>
    <tr>
    <th>5</th>
    <td>C</td>
    <td>5</td>
    <td>9</td>
    </tr>
    </tbody>
    
    
    
    
    <thead>
    <tr>
    <th></th>
    <th colspan="3" halign="left">data1</th>
    <th colspan="3" halign="left">data2</th>
    </tr>
    <tr>
    <th></th>
    <th>min</th>
    <th>median</th>
    <th>max</th>
    <th>min</th>
    <th>median</th>
    <th>max</th>
    </tr>
    <tr>
    <th>key</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th>A</th>
    <td>0</td>
    <td>1.5</td>
    <td>3</td>
    <td>3</td>
    <td>4.0</td>
    <td>5</td>
    </tr>
    <tr>
    <th>B</th>
    <td>1</td>
    <td>2.5</td>
    <td>4</td>
    <td>0</td>
    <td>3.5</td>
    <td>7</td>
    </tr>
    <tr>
    <th>C</th>
    <td>2</td>
    <td>3.5</td>
    <td>5</td>
    <td>3</td>
    <td>6.0</td>
    <td>9</td>
    </tr>
    </tbody>
    
    
    
    
    <thead>
    <tr style="text-align: right;">
    <th></th>
    <th>data2</th>
    <th>data1</th>
    </tr>
    <tr>
    <th>key</th>
    <th></th>
    <th></th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th>A</th>
    <td>5</td>
    <td>0</td>
    </tr>
    <tr>
    <th>B</th>
    <td>7</td>
    <td>1</td>
    </tr>
    <tr>
    <th>C</th>
    <td>9</td>
    <td>2</td>
    </tr>
    </tbody>
    
    
    
    
    <thead>
    <tr style="text-align: right;">
    <th></th>
    <th>key</th>
    <th>data1</th>
    <th>data2</th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th>1</th>
    <td>B</td>
    <td>1</td>
    <td>0</td>
    </tr>
    <tr>
    <th>2</th>
    <td>C</td>
    <td>2</td>
    <td>3</td>
    </tr>
    <tr>
    <th>4</th>
    <td>B</td>
    <td>4</td>
    <td>7</td>
    </tr>
    <tr>
    <th>5</th>
    <td>C</td>
    <td>5</td>
    <td>9</td>
    </tr>
    </tbody>
    
    
    
    
    <thead>
    <tr style="text-align: right;">
    <th></th>
    <th>data1</th>
    <th>data2</th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th>0</th>
    <td>-1.5</td>
    <td>1.0</td>
    </tr>
    <tr>
    <th>1</th>
    <td>-1.5</td>
    <td>-3.5</td>
    </tr>
    <tr>
    <th>2</th>
    <td>-1.5</td>
    <td>-3.0</td>
    </tr>
    <tr>
    <th>3</th>
    <td>1.5</td>
    <td>-1.0</td>
    </tr>
    <tr>
    <th>4</th>
    <td>1.5</td>
    <td>3.5</td>
    </tr>
    <tr>
    <th>5</th>
    <td>1.5</td>
    <td>3.0</td>
    </tr>
    </tbody>
    
    
    
    
    <thead>
    <tr style="text-align: right;">
    <th></th>
    <th>key</th>
    <th>data1</th>
    <th>data2</th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th>0</th>
    <td>A</td>
    <td>0.000000</td>
    <td>5</td>
    </tr>
    <tr>
    <th>1</th>
    <td>B</td>
    <td>0.142857</td>
    <td>0</td>
    </tr>
    <tr>
    <th>2</th>
    <td>C</td>
    <td>0.166667</td>
    <td>3</td>
    </tr>
    <tr>
    <th>3</th>
    <td>A</td>
    <td>0.375000</td>
    <td>3</td>
    </tr>
    <tr>
    <th>4</th>
    <td>B</td>
    <td>0.571429</td>
    <td>7</td>
    </tr>
    <tr>
    <th>5</th>
    <td>C</td>
    <td>0.416667</td>
    <td>9</td>
    </tr>
    </tbody>
    
    
    
    
    <thead>
    <tr style="text-align: right;">
    <th></th>
    <th>data1</th>
    <th>data2</th>
    </tr>
    <tr>
    <th>key</th>
    <th></th>
    <th></th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th>A</th>
    <td>3</td>
    <td>8</td>
    </tr>
    <tr>
    <th>B</th>
    <td>5</td>
    <td>7</td>
    </tr>
    <tr>
    <th>C</th>
    <td>7</td>
    <td>12</td>
    </tr>
    </tbody>
    
    
    
    
    <thead>
    <tr style="text-align: right;">
    <th></th>
    <th>data1</th>
    <th>data2</th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th>consonant</th>
    <td>12</td>
    <td>19</td>
    </tr>
    <tr>
    <th>vowel</th>
    <td>3</td>
    <td>8</td>
    </tr>
    </tbody>
    
    
    
    
    <thead>
    <tr style="text-align: right;">
    <th></th>
    <th>data1</th>
    <th>data2</th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th>a</th>
    <td>1.5</td>
    <td>4.0</td>
    </tr>
    <tr>
    <th>b</th>
    <td>2.5</td>
    <td>3.5</td>
    </tr>
    <tr>
    <th>c</th>
    <td>3.5</td>
    <td>6.0</td>
    </tr>
    </tbody>
    
    
    
    
    <thead>
    <tr style="text-align: right;">
    <th></th>
    <th></th>
    <th>data1</th>
    <th>data2</th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th>a</th>
    <th>vowel</th>
    <td>1.5</td>
    <td>4.0</td>
    </tr>
    <tr>
    <th>b</th>
    <th>consonant</th>
    <td>2.5</td>
    <td>3.5</td>
    </tr>
    <tr>
    <th>c</th>
    <th>consonant</th>
    <td>3.5</td>
    <td>6.0</td>
    </tr>
    </tbody>
    
    
    
    
    <thead>
    <tr style="text-align: right;">
    <th></th>
    <th>method</th>
    <th>number</th>
    <th>orbital_period</th>
    <th>mass</th>
    <th>distance</th>
    <th>year</th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th>0</th>
    <td>Radial Velocity</td>
    <td>1</td>
    <td>269.300</td>
    <td>7.10</td>
    <td>77.40</td>
    <td>2006</td>
    </tr>
    <tr>
    <th>1</th>
    <td>Radial Velocity</td>
    <td>1</td>
    <td>874.774</td>
    <td>2.21</td>
    <td>56.95</td>
    <td>2008</td>
    </tr>
    <tr>
    <th>2</th>
    <td>Radial Velocity</td>
    <td>1</td>
    <td>763.000</td>
    <td>2.60</td>
    <td>19.84</td>
    <td>2011</td>
    </tr>
    <tr>
    <th>3</th>
    <td>Radial Velocity</td>
    <td>1</td>
    <td>326.030</td>
    <td>19.40</td>
    <td>110.62</td>
    <td>2007</td>
    </tr>
    <tr>
    <th>4</th>
    <td>Radial Velocity</td>
    <td>1</td>
    <td>516.220</td>
    <td>10.50</td>
    <td>119.47</td>
    <td>2009</td>
    </tr>
    </tbody>
    
    
    
    
    <thead>
    <tr style="text-align: right;">
    <th>decade</th>
    <th>1980s</th>
    <th>1990s</th>
    <th>2000s</th>
    <th>2010s</th>
    </tr>
    <tr>
    <th>method</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th>Astrometry</th>
    <td>NaN</td>
    <td>NaN</td>
    <td>NaN</td>
    <td>2</td>
    </tr>
    <tr>
    <th>Eclipse Timing Variations</th>
    <td>NaN</td>
    <td>NaN</td>
    <td>5</td>
    <td>10</td>
    </tr>
    <tr>
    <th>Imaging</th>
    <td>NaN</td>
    <td>NaN</td>
    <td>29</td>
    <td>21</td>
    </tr>
    <tr>
    <th>Microlensing</th>
    <td>NaN</td>
    <td>NaN</td>
    <td>12</td>
    <td>15</td>
    </tr>
    <tr>
    <th>Orbital Brightness Modulation</th>
    <td>NaN</td>
    <td>NaN</td>
    <td>NaN</td>
    <td>5</td>
    </tr>
    <tr>
    <th>Pulsar Timing</th>
    <td>NaN</td>
    <td>9</td>
    <td>1</td>
    <td>1</td>
    </tr>
    <tr>
    <th>Pulsation Timing Variations</th>
    <td>NaN</td>
    <td>NaN</td>
    <td>1</td>
    <td>NaN</td>
    </tr>
    <tr>
    <th>Radial Velocity</th>
    <td>1</td>
    <td>52</td>
    <td>475</td>
    <td>424</td>
    </tr>
    <tr>
    <th>Transit</th>
    <td>NaN</td>
    <td>NaN</td>
    <td>64</td>
    <td>712</td>
    </tr>
    <tr>
    <th>Transit Timing Variations</th>
    <td>NaN</td>
    <td>NaN</td>
    <td>NaN</td>
    <td>9</td>
    </tr>
    </tbody>
    
    
    
    
    <thead>
    <tr style="text-align: right;">
    <th>decade</th>
    <th>1980s</th>
    <th>1990s</th>
    <th>2000s</th>
    <th>2010s</th>
    </tr>
    <tr>
    <th>method</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th>Astrometry</th>
    <td>NaN</td>
    <td>NaN</td>
    <td>NaN</td>
    <td>2</td>
    </tr>
    <tr>
    <th>Eclipse Timing Variations</th>
    <td>NaN</td>
    <td>NaN</td>
    <td>5</td>
    <td>10</td>
    </tr>
    <tr>
    <th>Imaging</th>
    <td>NaN</td>
    <td>NaN</td>
    <td>29</td>
    <td>21</td>
    </tr>
    <tr>
    <th>Microlensing</th>
    <td>NaN</td>
    <td>NaN</td>
    <td>12</td>
    <td>15</td>
    </tr>
    <tr>
    <th>Orbital Brightness Modulation</th>
    <td>NaN</td>
    <td>NaN</td>
    <td>NaN</td>
    <td>5</td>
    </tr>
    <tr>
    <th>Pulsar Timing</th>
    <td>NaN</td>
    <td>9</td>
    <td>1</td>
    <td>1</td>
    </tr>
    <tr>
    <th>Pulsation Timing Variations</th>
    <td>NaN</td>
    <td>NaN</td>
    <td>1</td>
    <td>NaN</td>
    </tr>
    <tr>
    <th>Radial Velocity</th>
    <td>1</td>
    <td>52</td>
    <td>475</td>
    <td>424</td>
    </tr>
    <tr>
    <th>Transit</th>
    <td>NaN</td>
    <td>NaN</td>
    <td>64</td>
    <td>712</td>
    </tr>
    <tr>
    <th>Transit Timing Variations</th>
    <td>NaN</td>
    <td>NaN</td>
    <td>NaN</td>
    <td>9</td>
    </tr>
    </tbody>
    
    
    
    
    <thead>
    <tr style="text-align: right;">
    <th>decade</th>
    <th>1980s</th>
    <th>1990s</th>
    <th>2000s</th>
    <th>2010s</th>
    </tr>
    <tr>
    <th>method</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th>Astrometry</th>
    <td>1</td>
    <td>30.5</td>
    <td>83.857143</td>
    <td>2.000000</td>
    </tr>
    <tr>
    <th>Eclipse Timing Variations</th>
    <td>1</td>
    <td>30.5</td>
    <td>5.000000</td>
    <td>10.000000</td>
    </tr>
    <tr>
    <th>Imaging</th>
    <td>1</td>
    <td>30.5</td>
    <td>29.000000</td>
    <td>21.000000</td>
    </tr>
    <tr>
    <th>Microlensing</th>
    <td>1</td>
    <td>30.5</td>
    <td>12.000000</td>
    <td>15.000000</td>
    </tr>
    <tr>
    <th>Orbital Brightness Modulation</th>
    <td>1</td>
    <td>30.5</td>
    <td>83.857143</td>
    <td>5.000000</td>
    </tr>
    <tr>
    <th>Pulsar Timing</th>
    <td>1</td>
    <td>9.0</td>
    <td>1.000000</td>
    <td>1.000000</td>
    </tr>
    <tr>
    <th>Pulsation Timing Variations</th>
    <td>1</td>
    <td>30.5</td>
    <td>1.000000</td>
    <td>133.222222</td>
    </tr>
    <tr>
    <th>Radial Velocity</th>
    <td>1</td>
    <td>52.0</td>
    <td>475.000000</td>
    <td>424.000000</td>
    </tr>
    <tr>
    <th>Transit</th>
    <td>1</td>
    <td>30.5</td>
    <td>64.000000</td>
    <td>712.000000</td>
    </tr>
    <tr>
    <th>Transit Timing Variations</th>
    <td>1</td>
    <td>30.5</td>
    <td>83.857143</td>
    <td>9.000000</td>
    </tr>
    </tbody>
    
    
    
    
    <thead>
    <tr style="text-align: right;">
    <th></th>
    <th>survived</th>
    <th>pclass</th>
    <th>sex</th>
    <th>age</th>
    <th>sibsp</th>
    <th>parch</th>
    <th>fare</th>
    <th>embarked</th>
    <th>class</th>
    <th>who</th>
    <th>adult_male</th>
    <th>deck</th>
    <th>embark_town</th>
    <th>alive</th>
    <th>alone</th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th>0</th>
    <td>0</td>
    <td>3</td>
    <td>male</td>
    <td>22</td>
    <td>1</td>
    <td>0</td>
    <td>7.2500</td>
    <td>S</td>
    <td>Third</td>
    <td>man</td>
    <td>True</td>
    <td>NaN</td>
    <td>Southampton</td>
    <td>no</td>
    <td>False</td>
    </tr>
    <tr>
    <th>1</th>
    <td>1</td>
    <td>1</td>
    <td>female</td>
    <td>38</td>
    <td>1</td>
    <td>0</td>
    <td>71.2833</td>
    <td>C</td>
    <td>First</td>
    <td>woman</td>
    <td>False</td>
    <td>C</td>
    <td>Cherbourg</td>
    <td>yes</td>
    <td>False</td>
    </tr>
    <tr>
    <th>2</th>
    <td>1</td>
    <td>3</td>
    <td>female</td>
    <td>26</td>
    <td>0</td>
    <td>0</td>
    <td>7.9250</td>
    <td>S</td>
    <td>Third</td>
    <td>woman</td>
    <td>False</td>
    <td>NaN</td>
    <td>Southampton</td>
    <td>yes</td>
    <td>True</td>
    </tr>
    <tr>
    <th>3</th>
    <td>1</td>
    <td>1</td>
    <td>female</td>
    <td>35</td>
    <td>1</td>
    <td>0</td>
    <td>53.1000</td>
    <td>S</td>
    <td>First</td>
    <td>woman</td>
    <td>False</td>
    <td>C</td>
    <td>Southampton</td>
    <td>yes</td>
    <td>False</td>
    </tr>
    <tr>
    <th>4</th>
    <td>0</td>
    <td>3</td>
    <td>male</td>
    <td>35</td>
    <td>0</td>
    <td>0</td>
    <td>8.0500</td>
    <td>S</td>
    <td>Third</td>
    <td>man</td>
    <td>True</td>
    <td>NaN</td>
    <td>Southampton</td>
    <td>no</td>
    <td>True</td>
    </tr>
    </tbody>
    
    
    
    
    <thead>
    <tr style="text-align: right;">
    <th></th>
    <th>survived</th>
    </tr>
    <tr>
    <th>sex</th>
    <th></th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th>female</th>
    <td>0.742038</td>
    </tr>
    <tr>
    <th>male</th>
    <td>0.188908</td>
    </tr>
    </tbody>
    
    
    
    
    <thead>
    <tr style="text-align: right;">
    <th>class</th>
    <th>First</th>
    <th>Second</th>
    <th>Third</th>
    </tr>
    <tr>
    <th>sex</th>
    <th></th>
    <th></th>
    <th></th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th>female</th>
    <td>0.968085</td>
    <td>0.921053</td>
    <td>0.500000</td>
    </tr>
    <tr>
    <th>male</th>
    <td>0.368852</td>
    <td>0.157407</td>
    <td>0.135447</td>
    </tr>
    </tbody>
    
    
    
    
    <thead>
    <tr style="text-align: right;">
    <th>class</th>
    <th>First</th>
    <th>Second</th>
    <th>Third</th>
    </tr>
    <tr>
    <th>sex</th>
    <th></th>
    <th></th>
    <th></th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th>female</th>
    <td>0.968085</td>
    <td>0.921053</td>
    <td>0.500000</td>
    </tr>
    <tr>
    <th>male</th>
    <td>0.368852</td>
    <td>0.157407</td>
    <td>0.135447</td>
    </tr>
    </tbody>
    
    
    
    
    <thead>
    <tr style="text-align: right;">
    <th></th>
    <th>class</th>
    <th>First</th>
    <th>Second</th>
    <th>Third</th>
    </tr>
    <tr>
    <th>sex</th>
    <th>age</th>
    <th></th>
    <th></th>
    <th></th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th rowspan="2" valign="top">female</th>
    <th>(0, 18]</th>
    <td>0.909091</td>
    <td>1.000000</td>
    <td>0.511628</td>
    </tr>
    <tr>
    <th>(18, 80]</th>
    <td>0.972973</td>
    <td>0.900000</td>
    <td>0.423729</td>
    </tr>
    <tr>
    <th rowspan="2" valign="top">male</th>
    <th>(0, 18]</th>
    <td>0.800000</td>
    <td>0.600000</td>
    <td>0.215686</td>
    </tr>
    <tr>
    <th>(18, 80]</th>
    <td>0.375000</td>
    <td>0.071429</td>
    <td>0.133663</td>
    </tr>
    </tbody>
    
    
    
    
    <thead>
    <tr>
    <th></th>
    <th>fare</th>
    <th colspan="3" halign="left">[0, 8.662]</th>
    <th colspan="3" halign="left">(8.662, 26]</th>
    <th colspan="3" halign="left">(26, 512.329]</th>
    </tr>
    <tr>
    <th></th>
    <th>class</th>
    <th>First</th>
    <th>Second</th>
    <th>Third</th>
    <th>First</th>
    <th>Second</th>
    <th>Third</th>
    <th>First</th>
    <th>Second</th>
    <th>Third</th>
    </tr>
    <tr>
    <th>sex</th>
    <th>age</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th rowspan="2" valign="top">female</th>
    <th>(0, 18]</th>
    <td>NaN</td>
    <td>NaN</td>
    <td>0.700000</td>
    <td>NaN</td>
    <td>1.000000</td>
    <td>0.583333</td>
    <td>0.909091</td>
    <td>1.0</td>
    <td>0.111111</td>
    </tr>
    <tr>
    <th>(18, 80]</th>
    <td>NaN</td>
    <td>NaN</td>
    <td>0.523810</td>
    <td>1</td>
    <td>0.877551</td>
    <td>0.433333</td>
    <td>0.972222</td>
    <td>1.0</td>
    <td>0.125000</td>
    </tr>
    <tr>
    <th rowspan="2" valign="top">male</th>
    <th>(0, 18]</th>
    <td>NaN</td>
    <td>NaN</td>
    <td>0.166667</td>
    <td>NaN</td>
    <td>0.500000</td>
    <td>0.500000</td>
    <td>0.800000</td>
    <td>0.8</td>
    <td>0.052632</td>
    </tr>
    <tr>
    <th>(18, 80]</th>
    <td>0</td>
    <td>NaN</td>
    <td>0.127389</td>
    <td>0</td>
    <td>0.086957</td>
    <td>0.102564</td>
    <td>0.400000</td>
    <td>0.0</td>
    <td>0.500000</td>
    </tr>
    </tbody>
    
    
    
    
    <thead>
    <tr>
    <th></th>
    <th colspan="3" halign="left">fare</th>
    <th colspan="3" halign="left">survived</th>
    </tr>
    <tr>
    <th>class</th>
    <th>First</th>
    <th>Second</th>
    <th>Third</th>
    <th>First</th>
    <th>Second</th>
    <th>Third</th>
    </tr>
    <tr>
    <th>sex</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th>female</th>
    <td>106.125798</td>
    <td>21.970121</td>
    <td>16.118810</td>
    <td>91</td>
    <td>70</td>
    <td>72</td>
    </tr>
    <tr>
    <th>male</th>
    <td>67.226127</td>
    <td>19.741782</td>
    <td>12.661633</td>
    <td>45</td>
    <td>17</td>
    <td>47</td>
    </tr>
    </tbody>
    
    
    
    
    <thead>
    <tr style="text-align: right;">
    <th>class</th>
    <th>First</th>
    <th>Second</th>
    <th>Third</th>
    <th>All</th>
    </tr>
    <tr>
    <th>sex</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th>female</th>
    <td>0.968085</td>
    <td>0.921053</td>
    <td>0.500000</td>
    <td>0.742038</td>
    </tr>
    <tr>
    <th>male</th>
    <td>0.368852</td>
    <td>0.157407</td>
    <td>0.135447</td>
    <td>0.188908</td>
    </tr>
    <tr>
    <th>All</th>
    <td>0.629630</td>
    <td>0.472826</td>
    <td>0.242363</td>
    <td>0.383838</td>
    </tr>
    </tbody>
    
    
    
    
    <thead>
    <tr style="text-align: right;">
    <th></th>
    <th>year</th>
    <th>month</th>
    <th>day</th>
    <th>gender</th>
    <th>births</th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th>0</th>
    <td>1969</td>
    <td>1</td>
    <td>1</td>
    <td>F</td>
    <td>4046</td>
    </tr>
    <tr>
    <th>1</th>
    <td>1969</td>
    <td>1</td>
    <td>1</td>
    <td>M</td>
    <td>4440</td>
    </tr>
    <tr>
    <th>2</th>
    <td>1969</td>
    <td>1</td>
    <td>2</td>
    <td>F</td>
    <td>4454</td>
    </tr>
    <tr>
    <th>3</th>
    <td>1969</td>
    <td>1</td>
    <td>2</td>
    <td>M</td>
    <td>4548</td>
    </tr>
    <tr>
    <th>4</th>
    <td>1969</td>
    <td>1</td>
    <td>3</td>
    <td>F</td>
    <td>4548</td>
    </tr>
    </tbody>
    
    
    
    
    <thead>
    <tr style="text-align: right;">
    <th>gender</th>
    <th>F</th>
    <th>M</th>
    </tr>
    <tr>
    <th>decade</th>
    <th></th>
    <th></th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th>1960</th>
    <td>1753634</td>
    <td>1846572</td>
    </tr>
    <tr>
    <th>1970</th>
    <td>16263075</td>
    <td>17121550</td>
    </tr>
    <tr>
    <th>1980</th>
    <td>18310351</td>
    <td>19243452</td>
    </tr>
    <tr>
    <th>1990</th>
    <td>19479454</td>
    <td>20420553</td>
    </tr>
    <tr>
    <th>2000</th>
    <td>18229309</td>
    <td>19106428</td>
    </tr>
    </tbody>
    
    
    
    
    <thead>
    <tr style="text-align: right;">
    <th></th>
    <th>year</th>
    <th>month</th>
    <th>day</th>
    <th>gender</th>
    <th>births</th>
    <th>decade</th>
    <th>dayofweek</th>
    </tr>
    </thead>
    
    
    <tbody>
    <tr>
    <th>1969-01-01</th>
    <td>1969</td>
    <td>1</td>
    <td>1</td>
    <td>F</td>
    <td>4046</td>
    <td>1960</td>
    <td>2</td>
    </tr>
    <tr>
    <th>1969-01-01</th>
    <td>1969</td>
    <td>1</td>
    <td>1</td>
    <td>M</td>
    <td>4440</td>
    <td>1960</td>
    <td>2</td>
    </tr>
    <tr>
    <th>1969-01-02</th>
    <td>1969</td>
    <td>1</td>
    <td>2</td>
    <td>F</td>
    <td>4454</td>
    <td>1960</td>
    <td>3</td>
    </tr>
    <tr>
    <th>1969-01-02</th>
    <td>1969</td>
    <td>1</td>
    <td>2</td>
    <td>M</td>
    <td>4548</td>
    <td>1960</td>
    <td>3</td>
    </tr>
    <tr>
    <th>1969-01-03</th>
    <td>1969</td>
    <td>1</td>
    <td>3</td>
    <td>F</td>
    <td>4548</td>
    <td>1960</td>
    <td>4</td>
    </tr>
    </tbody>
    
    



```python
for i in bsObj.find_all("table",{"class":"dataframe"}):
    for j in i.tr.next_siblings:
        print(j) 
```

    
    
    
    
    
    
    <tr>
    <th>method</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    </tr>
    
    
    
    
    
    
    <tr>
    <th></th>
    <th>min</th>
    <th>median</th>
    <th>max</th>
    <th>min</th>
    <th>median</th>
    <th>max</th>
    </tr>
    
    
    <tr>
    <th>key</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    </tr>
    
    
    
    
    <tr>
    <th>key</th>
    <th></th>
    <th></th>
    </tr>
    
    
    
    
    
    
    
    
    
    
    <tr>
    <th>key</th>
    <th></th>
    <th></th>
    </tr>
    
    
    
    
    
    
    
    
    
    
    
    
    <tr>
    <th>method</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    </tr>
    
    
    
    
    <tr>
    <th>method</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    </tr>
    
    
    
    
    <tr>
    <th>method</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    </tr>
    
    
    
    
    
    
    <tr>
    <th>sex</th>
    <th></th>
    </tr>
    
    
    
    
    <tr>
    <th>sex</th>
    <th></th>
    <th></th>
    <th></th>
    </tr>
    
    
    
    
    <tr>
    <th>sex</th>
    <th></th>
    <th></th>
    <th></th>
    </tr>
    
    
    
    
    <tr>
    <th>sex</th>
    <th>age</th>
    <th></th>
    <th></th>
    <th></th>
    </tr>
    
    
    
    
    <tr>
    <th></th>
    <th>class</th>
    <th>First</th>
    <th>Second</th>
    <th>Third</th>
    <th>First</th>
    <th>Second</th>
    <th>Third</th>
    <th>First</th>
    <th>Second</th>
    <th>Third</th>
    </tr>
    
    
    <tr>
    <th>sex</th>
    <th>age</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    </tr>
    
    
    
    
    <tr>
    <th>class</th>
    <th>First</th>
    <th>Second</th>
    <th>Third</th>
    <th>First</th>
    <th>Second</th>
    <th>Third</th>
    </tr>
    
    
    <tr>
    <th>sex</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    </tr>
    
    
    
    
    <tr>
    <th>sex</th>
    <th></th>
    <th></th>
    <th></th>
    <th></th>
    </tr>
    
    
    
    
    
    
    <tr>
    <th>decade</th>
    <th></th>
    <th></th>
    </tr>
    
    
    
    


find all my images in one post:


```python
import re
html = urlopen("https://karenyyyme.herokuapp.com/post/80/")
bsObj = BeautifulSoup(html, "html.parser")
images = bsObj.findAll("img", {"src":re.compile(".*\.png")})
for image in images: 
    print(image["src"])
```

    https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_2_0.png
    https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_4_0.png
    https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_7_0.png
    https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_11_0.png
    https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_16_0.png
    https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_18_0.png
    https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_22_0.png
    https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_24_0.png
    https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_28_0.png
    https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_30_0.png
    https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_36_0.png
    https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_38_0.png
    https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_42_0.png


lambda expressions:


```python
tags = bsObj.findAll(lambda tag: len(tag.attrs) == 2)
for tag in tags:
    print(tag)
```

    <meta content="width=device-width, initial-scale=1" name="viewport">
    <!-- css -->
    <link href="http://code.ionicframework.com/ionicons/2.0.1/css/ionicons.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css?family=Permanent+Marker" rel="stylesheet" type="text/css">
    <link href="https://fonts.googleapis.com/css?family=Raleway:100,600" rel="stylesheet" type="text/css">
    <link href="https://fonts.googleapis.com/css?family=Ubuntu+Mono" rel="stylesheet">
    <link href="/static/blog/css/bootstrap.min.css" rel="stylesheet">
    <link href="/static/blog/css/pace.css" rel="stylesheet">
    <link href="/static/blog/css/custom.css" rel="stylesheet">
    <link href="/static/blog/css/highlights/github.css" rel="stylesheet">
    <link href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css" rel="stylesheet">
    <link href="//cdnjs.cloudflare.com/ajax/libs/highlight.js/9.11.0/styles/default.min.css" rel="stylesheet">
    <style>
            span.highlighted {
                color: red;
            }
    
        </style>
    <!-- js -->
    <script src="/static/blog/js/jquery-2.1.3.min.js"></script>
    <script src="/static/blog/js/bootstrap.min.js"></script>
    <script src="/static/blog/js/pace.min.js"></script>
    <script src="/static/blog/js/modernizr.custom.js"></script>
    <script src="/static/MathJax/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
    <script src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script src="http://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.11.0/highlight.min.js"></script>
    </link></link></link></link></link></link></link></link></link></link></meta>
    <link href="http://code.ionicframework.com/ionicons/2.0.1/css/ionicons.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css?family=Permanent+Marker" rel="stylesheet" type="text/css">
    <link href="https://fonts.googleapis.com/css?family=Raleway:100,600" rel="stylesheet" type="text/css">
    <link href="https://fonts.googleapis.com/css?family=Ubuntu+Mono" rel="stylesheet">
    <link href="/static/blog/css/bootstrap.min.css" rel="stylesheet">
    <link href="/static/blog/css/pace.css" rel="stylesheet">
    <link href="/static/blog/css/custom.css" rel="stylesheet">
    <link href="/static/blog/css/highlights/github.css" rel="stylesheet">
    <link href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css" rel="stylesheet">
    <link href="//cdnjs.cloudflare.com/ajax/libs/highlight.js/9.11.0/styles/default.min.css" rel="stylesheet">
    <style>
            span.highlighted {
                color: red;
            }
    
        </style>
    <!-- js -->
    <script src="/static/blog/js/jquery-2.1.3.min.js"></script>
    <script src="/static/blog/js/bootstrap.min.js"></script>
    <script src="/static/blog/js/pace.min.js"></script>
    <script src="/static/blog/js/modernizr.custom.js"></script>
    <script src="/static/MathJax/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
    <script src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script src="http://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.11.0/highlight.min.js"></script>
    </link></link></link></link></link></link></link></link></link></link>
    <link href="https://fonts.googleapis.com/css?family=Ubuntu+Mono" rel="stylesheet">
    <link href="/static/blog/css/bootstrap.min.css" rel="stylesheet">
    <link href="/static/blog/css/pace.css" rel="stylesheet">
    <link href="/static/blog/css/custom.css" rel="stylesheet">
    <link href="/static/blog/css/highlights/github.css" rel="stylesheet">
    <link href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css" rel="stylesheet">
    <link href="//cdnjs.cloudflare.com/ajax/libs/highlight.js/9.11.0/styles/default.min.css" rel="stylesheet">
    <style>
            span.highlighted {
                color: red;
            }
    
        </style>
    <!-- js -->
    <script src="/static/blog/js/jquery-2.1.3.min.js"></script>
    <script src="/static/blog/js/bootstrap.min.js"></script>
    <script src="/static/blog/js/pace.min.js"></script>
    <script src="/static/blog/js/modernizr.custom.js"></script>
    <script src="/static/MathJax/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
    <script src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script src="http://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.11.0/highlight.min.js"></script>
    </link></link></link></link></link></link></link>
    <link href="/static/blog/css/bootstrap.min.css" rel="stylesheet">
    <link href="/static/blog/css/pace.css" rel="stylesheet">
    <link href="/static/blog/css/custom.css" rel="stylesheet">
    <link href="/static/blog/css/highlights/github.css" rel="stylesheet">
    <link href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css" rel="stylesheet">
    <link href="//cdnjs.cloudflare.com/ajax/libs/highlight.js/9.11.0/styles/default.min.css" rel="stylesheet">
    <style>
            span.highlighted {
                color: red;
            }
    
        </style>
    <!-- js -->
    <script src="/static/blog/js/jquery-2.1.3.min.js"></script>
    <script src="/static/blog/js/bootstrap.min.js"></script>
    <script src="/static/blog/js/pace.min.js"></script>
    <script src="/static/blog/js/modernizr.custom.js"></script>
    <script src="/static/MathJax/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
    <script src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script src="http://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.11.0/highlight.min.js"></script>
    </link></link></link></link></link></link>
    <link href="/static/blog/css/pace.css" rel="stylesheet">
    <link href="/static/blog/css/custom.css" rel="stylesheet">
    <link href="/static/blog/css/highlights/github.css" rel="stylesheet">
    <link href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css" rel="stylesheet">
    <link href="//cdnjs.cloudflare.com/ajax/libs/highlight.js/9.11.0/styles/default.min.css" rel="stylesheet">
    <style>
            span.highlighted {
                color: red;
            }
    
        </style>
    <!-- js -->
    <script src="/static/blog/js/jquery-2.1.3.min.js"></script>
    <script src="/static/blog/js/bootstrap.min.js"></script>
    <script src="/static/blog/js/pace.min.js"></script>
    <script src="/static/blog/js/modernizr.custom.js"></script>
    <script src="/static/MathJax/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
    <script src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script src="http://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.11.0/highlight.min.js"></script>
    </link></link></link></link></link>
    <link href="/static/blog/css/custom.css" rel="stylesheet">
    <link href="/static/blog/css/highlights/github.css" rel="stylesheet">
    <link href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css" rel="stylesheet">
    <link href="//cdnjs.cloudflare.com/ajax/libs/highlight.js/9.11.0/styles/default.min.css" rel="stylesheet">
    <style>
            span.highlighted {
                color: red;
            }
    
        </style>
    <!-- js -->
    <script src="/static/blog/js/jquery-2.1.3.min.js"></script>
    <script src="/static/blog/js/bootstrap.min.js"></script>
    <script src="/static/blog/js/pace.min.js"></script>
    <script src="/static/blog/js/modernizr.custom.js"></script>
    <script src="/static/MathJax/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
    <script src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script src="http://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.11.0/highlight.min.js"></script>
    </link></link></link></link>
    <link href="/static/blog/css/highlights/github.css" rel="stylesheet">
    <link href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css" rel="stylesheet">
    <link href="//cdnjs.cloudflare.com/ajax/libs/highlight.js/9.11.0/styles/default.min.css" rel="stylesheet">
    <style>
            span.highlighted {
                color: red;
            }
    
        </style>
    <!-- js -->
    <script src="/static/blog/js/jquery-2.1.3.min.js"></script>
    <script src="/static/blog/js/bootstrap.min.js"></script>
    <script src="/static/blog/js/pace.min.js"></script>
    <script src="/static/blog/js/modernizr.custom.js"></script>
    <script src="/static/MathJax/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
    <script src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script src="http://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.11.0/highlight.min.js"></script>
    </link></link></link>
    <link href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css" rel="stylesheet">
    <link href="//cdnjs.cloudflare.com/ajax/libs/highlight.js/9.11.0/styles/default.min.css" rel="stylesheet">
    <style>
            span.highlighted {
                color: red;
            }
    
        </style>
    <!-- js -->
    <script src="/static/blog/js/jquery-2.1.3.min.js"></script>
    <script src="/static/blog/js/bootstrap.min.js"></script>
    <script src="/static/blog/js/pace.min.js"></script>
    <script src="/static/blog/js/modernizr.custom.js"></script>
    <script src="/static/MathJax/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
    <script src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script src="http://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.11.0/highlight.min.js"></script>
    </link></link>
    <link href="//cdnjs.cloudflare.com/ajax/libs/highlight.js/9.11.0/styles/default.min.css" rel="stylesheet">
    <style>
            span.highlighted {
                color: red;
            }
    
        </style>
    <!-- js -->
    <script src="/static/blog/js/jquery-2.1.3.min.js"></script>
    <script src="/static/blog/js/bootstrap.min.js"></script>
    <script src="/static/blog/js/pace.min.js"></script>
    <script src="/static/blog/js/modernizr.custom.js"></script>
    <script src="/static/MathJax/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
    <script src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <script src="http://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.11.0/highlight.min.js"></script>
    </link>
    <script src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <nav class="main-nav" role="navigation">
    <div class="navbar-header">
    <button class="navbar-toggle" id="trigger-overlay" type="button">
    <span class="ion-navicon"></span>
    </button>
    </div>
    <div class="collapse navbar-collapse" id="bs-example-navbar-collapse-1">
    <ul class="nav navbar-nav navbar-right">
    <li class="cl-effect-11"><a data-hover="Home" href="/">Home</a></li>
    <li class="cl-effect-11"><a data-hover="Github" href="https://github.com/karenyyy">Github</a></li>
    </ul>
    </div><!-- /.navbar-collapse -->
    </nav>
    <div class="collapse navbar-collapse" id="bs-example-navbar-collapse-1">
    <ul class="nav navbar-nav navbar-right">
    <li class="cl-effect-11"><a data-hover="Home" href="/">Home</a></li>
    <li class="cl-effect-11"><a data-hover="Github" href="https://github.com/karenyyy">Github</a></li>
    </ul>
    </div>
    <a data-hover="Home" href="/">Home</a>
    <a data-hover="Github" href="https://github.com/karenyyy">Github</a>
    <a href="#" id="search-menu"><span class="ion-ios-search-strong" id="search-icon"></span></a>
    <span class="ion-ios-search-strong" id="search-icon"></span>
    <div class="search-form" id="search-form">
    <form action="/search/" id="searchform" method="get" role="search">
    <input name="q" placeholder="search" required="" type="search">
    <button type="submit"><span class="ion-ios-search-strong"></span></button>
    </input></form>
    </div>
    <time class="entry-date" datetime="Feb. 4, 2018, 4:22 a.m.">Feb. 4, 2018, 4:22 a.m.</time>
    <img alt="png" src="https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_2_0.png"/>
    <img alt="png" src="https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_4_0.png"/>
    <img alt="png" src="https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_7_0.png"/>
    <img alt="png" src="https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_11_0.png"/>
    <img alt="png" src="https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_16_0.png"/>
    <img alt="png" src="https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_18_0.png"/>
    <img alt="png" src="https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_22_0.png"/>
    <img alt="png" src="https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_24_0.png"/>
    <img alt="png" src="https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_28_0.png"/>
    <img alt="png" src="https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_30_0.png"/>
    <img alt="png" src="https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_36_0.png"/>
    <img alt="png" src="https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_38_0.png"/>
    <img alt="png" src="https://raw.githubusercontent.com/karenyyy/data_science/master/py_datasci/images2/output_42_0.png"/>


wikipedia scraping:


```python
import random, datetime
random.seed(datetime.datetime.now())

def getLinks(articleUrl):
    html = urlopen("http://en.wikipedia.org"+articleUrl)
    bsObj = BeautifulSoup(html, "html.parser")
    return bsObj.find("div", {"id":"bodyContent"}).findAll("a", href=re.compile("^(/wiki/)((?!:).)*$"))
links = getLinks("/wiki/Kevin_Bacon")
i=0
while len(links) > 0:
    i+=1
    if i==20:
        break
    newArticle = links[random.randint(0, len(links)-1)].attrs["href"]
    print(newArticle)
    links = getLinks(newArticle)
```

    /wiki/Diner_(film)
    /wiki/Jerry_Weintraub
    /wiki/The_Newton_Boys
    /wiki/Train_robbery
    /wiki/Payroll
    /wiki/Equal_opportunity
    /wiki/Brazil
    /wiki/Class_discrimination
    /wiki/South_End_Press
    /wiki/Brooklyn
    /wiki/Columbia_County,_New_York
    /wiki/Province_of_New_York
    /wiki/Tuscarora_(tribe)
    /wiki/Loyalist_(American_Revolution)
    /wiki/Andrew_Allen_(Pennsylvania)
    /wiki/Library_of_Congress_Control_Number
    /wiki/Library_catalog
    /wiki/WorldCat
    /wiki/IP_address



```python
pages = set()
def getLinks(pageUrl):
    global pages
    html = urlopen("http://en.wikipedia.org"+pageUrl)
    bsObj = BeautifulSoup(html, "html.parser")
    try:
        print(bsObj.h1.get_text())
        print(bsObj.find(id ="mw-content-text").findAll("p")[0])
        print(bsObj.find(id="ca-edit").find("span").find("a").attrs['href'])
    except AttributeError:
        print("This page is missing something! No worries though!")
    
    for link in bsObj.findAll("a", href=re.compile("^(/wiki/)")):
        if 'href' in link.attrs:
            if link.attrs['href'] not in pages:
                #We have encountered a new page
                newPage = link.attrs['href']
                print("----------------\n"+newPage)
                pages.add(newPage)
                getLinks(newPage)
getLinks("") 
```

    Main Page
    <p><b><a href="/wiki/Grand_Duchess_Olga_Alexandrovna_of_Russia" title="Grand Duchess Olga Alexandrovna of Russia">Grand Duchess Olga Alexandrovna of Russia</a></b> (1882–1960) was the youngest child of Emperor <a href="/wiki/Alexander_III_of_Russia" title="Alexander III of Russia">Alexander III of Russia</a> and younger sister of Emperor <a class="mw-redirect" href="/wiki/Nicholas_II" title="Nicholas II">Nicholas II</a>. Her father died when she was 12, and her brother Nicholas became emperor. At 19 she married <a href="/wiki/Duke_Peter_Alexandrovich_of_Oldenburg" title="Duke Peter Alexandrovich of Oldenburg">Duke Peter Alexandrovich of Oldenburg</a>; their marriage was unconsummated and was annulled by the Emperor in October 1916. The following month Olga married cavalry officer <a href="/wiki/Nikolai_Kulikovsky" title="Nikolai Kulikovsky">Nikolai Kulikovsky</a>, with whom she had fallen in love several years before. During the First World War, the Grand Duchess served as an army nurse at the front and was awarded a medal for personal gallantry. At the downfall of the <a class="mw-redirect" href="/wiki/Romanov" title="Romanov">Romanovs</a> in the <a class="mw-redirect" href="/wiki/Russian_Revolution_of_1917" title="Russian Revolution of 1917">Russian Revolution of 1917</a>, she fled to the <a href="/wiki/Crimea" title="Crimea">Crimea</a> with her husband and children, where they lived under the threat of assassination. After her brother and his family were <a class="mw-redirect" href="/wiki/Shooting_of_the_Romanov_family" title="Shooting of the Romanov family">shot by revolutionaries</a>, she and her family escaped to Denmark in February 1920. In exile, she was often sought out by <a href="/wiki/Romanov_impostors" title="Romanov impostors">Romanov impostors</a> who claimed to be her dead relatives. In 1948, feeling threatened by <a href="/wiki/Joseph_Stalin" title="Joseph Stalin">Joseph Stalin</a>'s regime, she emigrated with her immediate family to <a href="/wiki/Ontario" title="Ontario">Ontario</a>, Canada. (<a href="/wiki/Grand_Duchess_Olga_Alexandrovna_of_Russia" title="Grand Duchess Olga Alexandrovna of Russia"><b>Full article...</b></a>)</p>
    This page is missing something! No worries though!
    ----------------
    /wiki/Wikipedia
    Wikipedia
    <p><b>Wikipedia</b> (<span class="nowrap"><span class="IPA nopopups noexcerpt"><a href="/wiki/Help:IPA/English" title="Help:IPA/English">/<span style="border-bottom:1px dotted"><span title="/ˌ/: secondary stress follows">ˌ</span><span title="'w' in 'wind'">w</span><span title="/ɪ/: 'i' in 'kit'">ɪ</span><span title="'k' in 'kind'">k</span><span title="/ɪ/: 'i' in 'kit'">ɪ</span><span title="/ˈ/: primary stress follows">ˈ</span><span title="'p' in 'pie'">p</span><span title="/iː/: 'ee' in 'fleece'">iː</span><span title="'d' in 'dye'">d</span><span title="/i/: 'y' in 'happy'">i</span><span title="/ə/: 'a' in 'about'">ə</span></span>/</a></span><small class="nowrap"> (<span class="unicode haudio"><span class="fn"><span style="white-space:nowrap"><a href="/wiki/File:GT_Wikipedia_BE.ogg" title="About this sound"><img alt="About this sound" data-file-height="20" data-file-width="20" height="11" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Loudspeaker.svg/11px-Loudspeaker.svg.png" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Loudspeaker.svg/17px-Loudspeaker.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Loudspeaker.svg/22px-Loudspeaker.svg.png 2x" width="11"/></a> </span><a class="internal" href="//upload.wikimedia.org/wikipedia/commons/0/01/GT_Wikipedia_BE.ogg" title="GT Wikipedia BE.ogg">listen</a></span></span>)</small></span> <a href="/wiki/Help:Pronunciation_respelling_key" title="Help:Pronunciation respelling key"><i title="English pronunciation respelling"><span style="font-size:90%">WIK</span>-i-<span style="font-size:90%">PEE</span>-dee-ə</i></a> or <span class="nowrap"><span class="IPA nopopups noexcerpt"><a href="/wiki/Help:IPA/English" title="Help:IPA/English">/<span style="border-bottom:1px dotted"><span title="/ˌ/: secondary stress follows">ˌ</span><span title="'w' in 'wind'">w</span><span title="/ɪ/: 'i' in 'kit'">ɪ</span><span title="'k' in 'kind'">k</span><span title="/i/: 'y' in 'happy'">i</span><span title="/ˈ/: primary stress follows">ˈ</span><span title="'p' in 'pie'">p</span><span title="/iː/: 'ee' in 'fleece'">iː</span><span title="'d' in 'dye'">d</span><span title="/i/: 'y' in 'happy'">i</span><span title="/ə/: 'a' in 'about'">ə</span></span>/</a></span><small class="nowrap"> (<span class="unicode haudio"><span class="fn"><span style="white-space:nowrap"><a href="/wiki/File:GT_Wikipedia_AE.ogg" title="About this sound"><img alt="About this sound" data-file-height="20" data-file-width="20" height="11" src="//upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Loudspeaker.svg/11px-Loudspeaker.svg.png" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Loudspeaker.svg/17px-Loudspeaker.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/8/8a/Loudspeaker.svg/22px-Loudspeaker.svg.png 2x" width="11"/></a> </span><a class="internal" href="//upload.wikimedia.org/wikipedia/commons/4/4c/GT_Wikipedia_AE.ogg" title="GT Wikipedia AE.ogg">listen</a></span></span>)</small></span> <a href="/wiki/Help:Pronunciation_respelling_key" title="Help:Pronunciation respelling key"><i title="English pronunciation respelling"><span style="font-size:90%">WIK</span>-ee-<span style="font-size:90%">PEE</span>-dee-ə</i></a>) is a free <a href="/wiki/Online_encyclopedia" title="Online encyclopedia">online encyclopedia</a> with the mission of allowing anyone to create or edit articles.<sup class="reference" id="cite_ref-6"><a href="#cite_note-6">[3]</a></sup><sup class="noprint Inline-Template" style="white-space:nowrap;">[<i><a href="/wiki/Wikipedia:Verifiability" title="Wikipedia:Verifiability"><span title="The material near this tag failed verification of its source citation(s). (January 2018)">not in citation given</span></a></i>]</sup> Wikipedia is the largest and most popular general <a href="/wiki/Reference_work" title="Reference work">reference work</a> on the Internet,<sup class="reference" id="cite_ref-Tancer_7-0"><a href="#cite_note-Tancer-7">[4]</a></sup><sup class="reference" id="cite_ref-Woodson_8-0"><a href="#cite_note-Woodson-8">[5]</a></sup><sup class="reference" id="cite_ref-9"><a href="#cite_note-9">[6]</a></sup> and is ranked the fifth-most popular website.<sup class="reference" id="cite_ref-Alexa_siteinfo_10-0"><a href="#cite_note-Alexa_siteinfo-10">[7]</a></sup> Wikipedia is a project owned by the nonprofit <a href="/wiki/Wikimedia_Foundation" title="Wikimedia Foundation">Wikimedia Foundation</a>.<sup class="reference" id="cite_ref-11"><a href="#cite_note-11">[8]</a></sup><sup class="reference" id="cite_ref-12"><a href="#cite_note-12">[9]</a></sup><sup class="reference" id="cite_ref-13"><a href="#cite_note-13">[10]</a></sup></p>
    This page is missing something! No worries though!
    ----------------
    /wiki/Wikipedia:Protection_policy#semi
    Wikipedia:Protection policy
    <p>Wikipedia is built around the principle that <a href="/wiki/Wiki" title="Wiki">anyone can edit it</a>, and it therefore aims to have as many of its pages as possible open for public editing so that anyone can add material and correct errors. However, in some particular circumstances, because of a specifically identified likelihood of damage resulting if editing is left open, some individual pages may need to be subject to technical restrictions (often only temporary but sometimes indefinitely) on who is permitted to modify them. The placing of such restrictions on pages is called <b>protection</b>.</p>
    This page is missing something! No worries though!
    ----------------
    /wiki/Wikipedia:Requests_for_page_protection
    Wikipedia:Requests for page protection
    <p>This page is for requesting that a page, file or template be <b>fully protected</b>, <b>create protected</b> (<a href="/wiki/Wikipedia:Protection_policy#Creation_protection" title="Wikipedia:Protection policy">salted</a>), <b>extended confirmed protected</b>, <b>semi-protected</b>, added to <b>pending changes</b>, <b>move-protected</b>, <b>template protected</b> (template-specific), <b>upload protected</b> (file-specific), or <b>unprotected</b>. Please read up on the <a href="/wiki/Wikipedia:Protection_policy" title="Wikipedia:Protection policy">protection policy</a>. Full protection is used to stop edit warring between multiple users or to prevent vandalism to <a href="/wiki/Wikipedia:High-risk_templates" title="Wikipedia:High-risk templates">high-risk templates</a>; semi-protection and pending changes are usually used only to prevent IP and new user vandalism (see the <a href="/wiki/Wikipedia:Rough_guide_to_semi-protection" title="Wikipedia:Rough guide to semi-protection">rough guide to semi-protection</a>); and move protection is used to stop <a href="/wiki/Wikipedia:Moving_a_page" title="Wikipedia:Moving a page">pagemove</a> revert wars. Extended confirmed protection is used where semi-protection has proved insufficient (see the <a href="/wiki/Wikipedia:Rough_guide_to_extended_confirmed_protection" title="Wikipedia:Rough guide to extended confirmed protection">rough guide to extended confirmed protection</a>)</p>
    /w/index.php?title=Wikipedia:Requests_for_page_protection&action=edit
    ----------------
    /wiki/Wikipedia:Requests_for_permissions
    Wikipedia:Requests for permissions
    <p><span class="sysop-show" id="coordinates"><a href="/wiki/Wikipedia:Requests_for_permissions/Administrator_instructions" title="Wikipedia:Requests for permissions/Administrator instructions">Administrator instructions</a></span></p>
    This page is missing something! No worries though!
    ----------------
    /wiki/Wikipedia:Requesting_copyright_permission
    Wikipedia:Requesting copyright permission
    <p>To use copyrighted material on Wikipedia, it is <i>not enough</i> that we have permission to use it on Wikipedia alone. That's because Wikipedia itself states all its material may be used by anyone, for any purpose. So we have to be sure all material is in fact licensed for that purpose, whoever provided it.</p>
    This page is missing something! No worries though!
    ----------------
    /wiki/Wikipedia:User_access_levels
    Wikipedia:User access levels
    <p>The <b>user access level</b> of an editor affects their ability to perform certain actions on Wikipedia; it depends on which <i>rights</i> (also called <i>permissions</i>, <i><a href="/wiki/Internet_forum#User_groups" title="Internet forum">user groups</a></i>, <a class="mw-redirect" href="/wiki/Bit_(computing)" title="Bit (computing)"><i>bits</i></a> or <a class="mw-redirect" href="/wiki/Flag_(computing)" title="Flag (computing)"><i>flags</i></a>) are assigned to accounts. This is determined by whether the editor is <a class="mw-redirect" href="/wiki/Wikipedia:Logging_in" title="Wikipedia:Logging in">logged into</a> an account, and whether the account has a sufficient age and number of edits for certain automatic rights, and what additional rights have been assigned manually to the account.</p>
    This page is missing something! No worries though!
    ----------------
    /wiki/Wikipedia:Requests_for_adminship
    Wikipedia:Requests for adminship
    <p><input class="mw-inputbox-input searchboxInput mw-ui-input mw-ui-input-inline" dir="ltr" name="search" placeholder="" size="30" type="text" value=""/><input name="prefix" type="hidden" value="Wikipedia:Requests for adminship/"/><br/>
    <input class="mw-ui-button" name="fulltext" type="submit" value="Search RfA"/><input name="fulltext" type="hidden" value="Search"/></p>
    This page is missing something! No worries though!
    ----------------
    /wiki/Wikipedia:Protection_policy#extended
    Wikipedia:Protection policy
    <p>Wikipedia is built around the principle that <a href="/wiki/Wiki" title="Wiki">anyone can edit it</a>, and it therefore aims to have as many of its pages as possible open for public editing so that anyone can add material and correct errors. However, in some particular circumstances, because of a specifically identified likelihood of damage resulting if editing is left open, some individual pages may need to be subject to technical restrictions (often only temporary but sometimes indefinitely) on who is permitted to modify them. The placing of such restrictions on pages is called <b>protection</b>.</p>
    This page is missing something! No worries though!
    ----------------
    /wiki/Wikipedia:Lists_of_protected_pages
    Wikipedia:Lists of protected pages
    <p>This is a list of resources available that list pages that are <b><a href="/wiki/Wikipedia:Protection_policy" title="Wikipedia:Protection policy">protected</a></b>, <b><a href="/wiki/Wikipedia:Protection_policy#Semi-protection" title="Wikipedia:Protection policy">semi-protected</a></b> or <b><a href="/wiki/Wikipedia:Protection_policy#Move_protected" title="Wikipedia:Protection policy">move protected</a>.</b></p>
    This page is missing something! No worries though!
    ----------------
    /wiki/Wikipedia:Protection_policy
    Wikipedia:Protection policy
    <p>Wikipedia is built around the principle that <a href="/wiki/Wiki" title="Wiki">anyone can edit it</a>, and it therefore aims to have as many of its pages as possible open for public editing so that anyone can add material and correct errors. However, in some particular circumstances, because of a specifically identified likelihood of damage resulting if editing is left open, some individual pages may need to be subject to technical restrictions (often only temporary but sometimes indefinitely) on who is permitted to modify them. The placing of such restrictions on pages is called <b>protection</b>.</p>
    This page is missing something! No worries though!
    ----------------
    /wiki/Wikipedia:Perennial_proposals
    Wikipedia:Perennial proposals
    <p>This is a list of things that are frequently proposed on Wikipedia, and have been <a class="mw-redirect" href="/wiki/Wikipedia:Rejected_proposals" title="Wikipedia:Rejected proposals">rejected by the community</a> several times in the past. It should be noted that merely listing something on this page does not mean it will never happen, but that it has been discussed before and never met consensus. <a class="mw-redirect" href="/wiki/Wikipedia:Consensus_can_change" title="Wikipedia:Consensus can change">Consensus can change</a>, and some proposals that remained on this page for a long time have finally been proposed in a way that reached consensus, but you should address rebuttals raised in the past if you make a proposal along these lines. If you feel you would still like to do one of these proposals, then raise it at the <a href="/wiki/Wikipedia:Village_pump" title="Wikipedia:Village pump">Wikipedia:Village pump</a>.</p>
    /w/index.php?title=Wikipedia:Perennial_proposals&action=edit
    ----------------
    /wiki/Wikipedia:Project_namespace#How-to_and_information_pages
    Wikipedia:Project namespace
    <p>Project pages are for information or discussion about Wikipedia. They should be used to allow Wikipedians to better participate in the community, and not used to excess for unrelated purposes nor to bring the project into disrepute.</p>
    This page is missing something! No worries though!
    ----------------
    /wiki/Wikipedia:Protection_policy#move
    Wikipedia:Protection policy
    <p>Wikipedia is built around the principle that <a href="/wiki/Wiki" title="Wiki">anyone can edit it</a>, and it therefore aims to have as many of its pages as possible open for public editing so that anyone can add material and correct errors. However, in some particular circumstances, because of a specifically identified likelihood of damage resulting if editing is left open, some individual pages may need to be subject to technical restrictions (often only temporary but sometimes indefinitely) on who is permitted to modify them. The placing of such restrictions on pages is called <b>protection</b>.</p>
    This page is missing something! No worries though!
    ----------------
    /wiki/Wikipedia:WPPP
    Wikipedia:WikiProject Parliamentary Procedure
    <p><b>WikiProject Parliamentary Procedure</b> is devoted to improving the quality and comprehensiveness of articles on topics related to <a href="/wiki/Parliamentary_procedure" title="Parliamentary procedure">parliamentary procedure</a>.</p>
    /w/index.php?title=Wikipedia:WikiProject_Parliamentary_Procedure&action=edit
    ----------------
    /wiki/File:People_icon.svg
    File:People icon.svg
    <p>People icon</p>
    This page is missing something! No worries though!
    ----------------
    /wiki/Special:WhatLinksHere/File:People_icon.svg
    Pages that link to "File:People icon.svg"
    <p>The following pages link to <b><span id="specialDeleteTarget"><a href="/wiki/File:People_icon.svg" title="File:People icon.svg">File:People icon.svg</a></span></b>
    <span id="specialDeleteLink"></span>
    </p>
    This page is missing something! No worries though!
    ----------------
    /wiki/Help:What_links_here
    Help:What links here
    <p>Within the Toolbox section on the left-hand side of every page is a link labeled "<b>What links here</b>". This is used to see a list of the pages that <a href="/wiki/Help:Link" title="Help:Link">link</a> to (or <a href="/wiki/Wikipedia:Redirect" title="Wikipedia:Redirect">redirect</a> to, or <a href="/wiki/Wikipedia:Transclusion" title="Wikipedia:Transclusion">transclude</a>) the current page. These are sometimes referred to as <b><a href="/wiki/Backlink" title="Backlink">backlinks</a></b>.</p>
    This page is missing something! No worries though!
    ----------------
    /wiki/Wikipedia:Policies_and_guidelines
    Wikipedia:Policies and guidelines
    <p>Wikipedia <b>policies and guidelines</b> are developed by the community to describe best practices, clarify principles, resolve conflicts, and otherwise further our goal of creating a free, reliable encyclopedia. There is no need to read any policy or guideline pages to start editing. The <a href="/wiki/Wikipedia:Five_pillars" title="Wikipedia:Five pillars">five pillars</a> is a popular summary of the most pertinent principles.</p>
    This page is missing something! No worries though!
    ----------------
    /wiki/Wikipedia:Shortcut
    Wikipedia:Shortcut
    <p>A <b>shortcut</b> is a specialized type of <a href="/wiki/Wikipedia:Redirect" title="Wikipedia:Redirect">redirect page</a> that provides an abbreviated <a class="mw-redirect" href="/wiki/Wikilink" title="Wikilink">wikilink</a> to a project page or one of its sections, usually from the <b><a href="/wiki/Wikipedia:Project_namespace" title="Wikipedia:Project namespace">Wikipedia namespace</a></b> and <b><a href="/wiki/Wikipedia:Help_namespace" title="Wikipedia:Help namespace">Help namespace</a></b>. They are commonly used on community pages and talk pages, but should not be used in articles themselves. If there is a shortcut for a page or section, it is usually displayed in an information box labelled <i>Shortcuts:</i>, as can be seen at the top of this page.</p>
    This page is missing something! No worries though!
    ----------------
    /wiki/Wikipedia:Keyboard_shortcuts
    Wikipedia:Keyboard shortcuts
    <p>The <a href="/wiki/MediaWiki" title="MediaWiki">MediaWiki</a> software contains many <a href="/wiki/Keyboard_shortcut" title="Keyboard shortcut">keyboard shortcuts</a>. You can use them to access certain features of Wikipedia more quickly.</p>
    This page is missing something! No worries though!
    ----------------
    /wiki/Wikipedia:WikiProject_Kansas
    Wikipedia:WikiProject Kansas
    <p><span style="font-size:100%;font-weight:bold;border: none; margin: 0; padding:0; padding-bottom:.1em; color:#FFD700;"><a class="image" href="/wiki/File:Seal_of_Kansas.svg"><img alt="Seal of Kansas.svg" data-file-height="600" data-file-width="600" height="48" src="//upload.wikimedia.org/wikipedia/commons/thumb/4/45/Seal_of_Kansas.svg/48px-Seal_of_Kansas.svg.png" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/4/45/Seal_of_Kansas.svg/72px-Seal_of_Kansas.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/4/45/Seal_of_Kansas.svg/96px-Seal_of_Kansas.svg.png 2x" width="48"/></a><br/>
    <i>Welcome</i></span></p>
    /w/index.php?title=Wikipedia:WikiProject_Kansas&action=edit
    ----------------
    /wiki/Wikipedia:WikiProject
    Wikipedia:WikiProject
    <p><input class="mw-inputbox-input searchboxInput mw-ui-input mw-ui-input-inline" dir="ltr" name="search" placeholder="" size="15" type="text" value=""/><input name="prefix" type="hidden" value="Wikipedia:WikiProject"/><br/>
    <input class="mw-ui-button" name="fulltext" type="submit" value="Search"/><input name="fulltext" type="hidden" value="Search"/></p>
    This page is missing something! No worries though!
    ----------------
    /wiki/Wikipedia:Wikimedia_sister_projects
    Wikipedia:Wikimedia sister projects
    <p><b>Wikimedia sister projects</b> are all the publicly available <a href="/wiki/Wiki" title="Wiki">wikis</a> operated by the <a href="/wiki/Wikimedia_Foundation" title="Wikimedia Foundation">Wikimedia Foundation</a>, including Wikipedia. This guideline covers Wikipedia's relations to the sister projects, including linking and copying content between a Wikipedia article and a sister project's article.</p>
    This page is missing something! No worries though!
    ----------------
    /wiki/Help:Interwikimedia_links
    Help:Interwikimedia links
    <p>An <b>interwikimedia link</b> is a <a class="mw-redirect" href="/wiki/Help:Contents/Links" title="Help:Contents/Links">link</a> from a <a href="/wiki/Wikipedia:Wikimedia_Foundation" title="Wikipedia:Wikimedia Foundation">Wikimedia Foundation</a>-supported project to another one (e.g., from <a href="/wiki/Wikipedia" title="Wikipedia">Wikipedia</a> to <a href="/wiki/Wiktionary" title="Wiktionary">Wiktionary</a>).</p>
    /w/index.php?title=Help:Interwikimedia_links&action=edit
    ----------------
    /wiki/Help:Interlanguage_links
    Help:Interlanguage links
    <p><b>Interlanguage links</b> are links from a page in one <a href="/wiki/Wikipedia" title="Wikipedia">Wikipedia</a> language to an equivalent page in another language. These links can appear in two places:</p>
    This page is missing something! No worries though!
    ----------------
    /wiki/List_of_ISO_639-1_codes
    List of ISO 639-1 codes
    <p><a href="/wiki/ISO_639" title="ISO 639">ISO 639</a> is a standardized nomenclature used to classify <a href="/wiki/Language" title="Language">languages</a>. Each language is assigned a two-letter (639-1) and three-letter (639-2 and 639-3), lowercase abbreviation, amended in later versions of the nomenclature. The system is highly useful for linguists and ethnographers to categorize the languages spoken on a regional basis, and to compute analysis in the field of <a href="/wiki/Lexicostatistics" title="Lexicostatistics">lexicostatistics</a>.</p>
    /w/index.php?title=List_of_ISO_639-1_codes&action=edit
    ----------------
    /wiki/File:Question_book-new.svg
    File:Question book-new.svg
    <p><b>English:</b>  A new incarnation of <a href="/wiki/File:Question_book-3.svg" title="File:Question book-3.svg">Image:Question_book-3.svg</a>, which was uploaded by user <a href="/wiki/User:AzaToth" title="User:AzaToth">AzaToth</a></p>
    /w/index.php?title=File:Question_book-new.svg&action=edit
    ----------------
    /wiki/Wikipedia:Protection_policy#full
    Wikipedia:Protection policy
    <p>Wikipedia is built around the principle that <a href="/wiki/Wiki" title="Wiki">anyone can edit it</a>, and it therefore aims to have as many of its pages as possible open for public editing so that anyone can add material and correct errors. However, in some particular circumstances, because of a specifically identified likelihood of damage resulting if editing is left open, some individual pages may need to be subject to technical restrictions (often only temporary but sometimes indefinitely) on who is permitted to modify them. The placing of such restrictions on pages is called <b>protection</b>.</p>
    This page is missing something! No worries though!
    ----------------
    /wiki/Wikipedia:Child_protection
    Wikipedia:Child protection
    <p>Wikipedia regards the safety of children using the site as a key issue. Editors who attempt to use Wikipedia to pursue or facilitate inappropriate adult–child relationships, who advocate inappropriate adult–child relationships on- or off-wiki (e.g. by expressing the view that inappropriate relationships are not harmful to children), or who identify themselves as <a href="/wiki/Pedophilia" title="Pedophilia">pedophiles</a>, will be <a class="mw-redirect" href="/wiki/Wikipedia:BLOCK" title="Wikipedia:BLOCK">blocked</a> indefinitely.</p>
    /w/index.php?title=Wikipedia:Child_protection&action=edit
    ----------------
    /wiki/Wikipedia:Biographies_of_living_people
    Wikipedia:Biographies of living persons
    <p>Editors must take particular care when adding <b>information about living persons</b> to <i>any</i> Wikipedia page.<sup class="reference" id="cite_ref-1"><a href="#cite_note-1">[a]</a></sup> Such material requires a high degree of sensitivity, and must adhere <i>strictly</i> to all applicable laws in the United States, to this policy, and to Wikipedia's three core content policies:</p>
    This page is missing something! No worries though!
    ----------------
    /wiki/Wikipedia:Biographies_of_living_persons/Noticeboard
    Wikipedia:Biographies of living persons/Noticeboard
    <p><input class="mw-inputbox-input searchboxInput mw-ui-input mw-ui-input-inline" dir="ltr" name="search" placeholder="" size="40" type="text" value=""/><input name="prefix" type="hidden" value="Wikipedia:Biographies of living persons/Noticeboard/"/><br/>
    <input class="mw-ui-button" name="fulltext" type="submit" value="Search"/><input name="fulltext" type="hidden" value="Search"/></p>
    /w/index.php?title=Wikipedia:Biographies_of_living_persons/Noticeboard&action=edit
    ----------------
    /wiki/Wikipedia:Biographies_of_living_persons
    Wikipedia:Biographies of living persons
    <p>Editors must take particular care when adding <b>information about living persons</b> to <i>any</i> Wikipedia page.<sup class="reference" id="cite_ref-1"><a href="#cite_note-1">[a]</a></sup> Such material requires a high degree of sensitivity, and must adhere <i>strictly</i> to all applicable laws in the United States, to this policy, and to Wikipedia's three core content policies:</p>
    This page is missing something! No worries though!
    ----------------
    /wiki/Wikipedia:What_%22Ignore_all_rules%22_means#Use_common_sense
    Wikipedia:What "Ignore all rules" means
    <p><br/></p>
    This page is missing something! No worries though!
    ----------------
    /wiki/Wikipedia:Ignore_all_rules
    Wikipedia:Ignore all rules
    <p>If a <a href="/wiki/Wikipedia:Policies_and_guidelines" title="Wikipedia:Policies and guidelines">rule</a> prevents you from improving or maintaining <a href="/wiki/Wikipedia" title="Wikipedia">Wikipedia</a>, <b>ignore it</b>.</p>
    /w/index.php?title=Wikipedia:Ignore_all_rules&action=edit
    ----------------
    /wiki/Wikipedia:Protection_policy#pc1
    Wikipedia:Protection policy
    <p>Wikipedia is built around the principle that <a href="/wiki/Wiki" title="Wiki">anyone can edit it</a>, and it therefore aims to have as many of its pages as possible open for public editing so that anyone can add material and correct errors. However, in some particular circumstances, because of a specifically identified likelihood of damage resulting if editing is left open, some individual pages may need to be subject to technical restrictions (often only temporary but sometimes indefinitely) on who is permitted to modify them. The placing of such restrictions on pages is called <b>protection</b>.</p>
    This page is missing something! No worries though!
    ----------------
    /wiki/Wikipedia:WikiProject_Protected_areas
    Wikipedia:WikiProject Protected areas
    <p>This <a href="/wiki/Wikipedia:WikiProject" title="Wikipedia:WikiProject">WikiProject</a> aims primarily to provide information on all <b><a href="/wiki/Protected_area" title="Protected area">protected areas</a></b> (abbreviated on this page as <b>PAs</b>) in the world.</p>
    /w/index.php?title=Wikipedia:WikiProject_Protected_areas&action=edit
    ----------------
    /wiki/Wikipedia:WikiProject_Council/Guide
    Wikipedia:WikiProject Council/Guide
    <p><input class="mw-inputbox-input searchboxInput mw-ui-input mw-ui-input-inline" dir="ltr" name="search" placeholder="" size="15" type="text" value=""/><input name="prefix" type="hidden" value="Wikipedia:WikiProject"/><br/>
    <input class="mw-ui-button" name="fulltext" type="submit" value="Search"/><input name="fulltext" type="hidden" value="Search"/></p>
    This page is missing something! No worries though!
    ----------------
    /wiki/Wikipedia:WikiProject_Council
    Wikipedia:WikiProject Council
    <p>The <b>WikiProject Council</b> is a group of Wikipedians gathered to encourage—and assist with—the development of active <a href="/wiki/Wikipedia:WikiProject" title="Wikipedia:WikiProject">WikiProjects</a>, and to act as a central point for inter-WikiProject discussion and collaboration. New to WikiProjects? See <a href="/wiki/Wikipedia:Wikipedia_Signpost/2013-04-01/WikiProject_report" title="Wikipedia:Wikipedia Signpost/2013-04-01/WikiProject report">FAQs about WikiProjects</a> for the answers to the most common inquiries. The <a href="/wiki/Wikipedia:WikiProject_Council/Guide" title="Wikipedia:WikiProject Council/Guide">WikiProject guideline page</a> outlines generally accepted protocols and conventions.</p>
    /w/index.php?title=Wikipedia:WikiProject_Council&action=edit
    ----------------
    /wiki/Wikipedia:Wikipedia_Signpost/2013-04-01/WikiProject_report
    Wikipedia:Wikipedia Signpost/2013-04-01/WikiProject report
    <p>Instead of interviewing a WikiProject, this week's Report is dedicated to answering our readers' questions about WikiProjects. The following Frequently Asked Questions came from feedback at the WikiProject Report's talk page, the WikiProject Council's talk page, and from previous lists of FAQs. Included in today's Report are questions and answers that may prove useful to Wikipedia's newest editors as well as seasoned veterans.</p>
    /w/index.php?title=Wikipedia:Wikipedia_Signpost/2013-04-01/WikiProject_report&action=edit
    ----------------
    /wiki/Wikipedia:Wikipedia_Signpost
    Wikipedia:Wikipedia Signpost
    <p><span style="font-size:90%;">Should an editor's block history be a permanent "rap sheet", or does Wikipedia forgive <i>and</i> forget? A reform initiative has begun.</span> <span class="autocomment">(<a href="/wiki/Wikipedia:Wikipedia_Signpost/2018-02-05/Op-ed" title="Wikipedia:Wikipedia Signpost/2018-02-05/Op-ed">continued→</a>)</span></p>
    This page is missing something! No worries though!
    ----------------
    /wiki/Wikipedia:Wikipedia_Signpost/2018-02-05/Op-ed
    Wikipedia:Wikipedia Signpost/2018-02-05/Op-ed
    <p><br/></p>
    /w/index.php?title=Wikipedia:Wikipedia_Signpost/2018-02-05/Op-ed&action=edit
    ----------------
    /wiki/Wikipedia:Wikipedia_Signpost/2018-02-05
    Wikipedia:Wikipedia Signpost/2018-02-05



    ---------------------------------------------------------------------------

    IndexError                                Traceback (most recent call last)

    <ipython-input-35-547bc4e8bdb1> in <module>()
         19                 pages.add(newPage)
         20                 getLinks(newPage)
    ---> 21 getLinks("")
    

    <ipython-input-35-547bc4e8bdb1> in getLinks(pageUrl)
         18                 print("----------------\n"+newPage)
         19                 pages.add(newPage)
    ---> 20                 getLinks(newPage)
         21 getLinks("")


    <ipython-input-35-547bc4e8bdb1> in getLinks(pageUrl)
         18                 print("----------------\n"+newPage)
         19                 pages.add(newPage)
    ---> 20                 getLinks(newPage)
         21 getLinks("")


 


    IndexError: list index out of range


> How does 'os.sep' works?

__os.sep = '/'__


```python
from os import sep
link="https://media.giphy.com/media/piKXr2hEDsO1G/giphy.gif"
link.strip(sep).rsplit(sep, 1)[-1]
```




    '/'



### scrape anime pics


```python
import requests
import os
import traceback

def download(url, filename):
    if os.path.exists(filename):
        print('')
        return
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=256):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    f.flush()
        return filename
    except KeyboardInterrupt:
        if os.path.exists(filename):
            os.remove(filename)
        raise KeyboardInterrupt
    except Exception:
        traceback.print_exc()
        if os.path.exists(filename):
            os.remove(filename)

```


```python
if os.path.exists('imgs') is False:
    os.makedirs('imgs')

start = 1
end = 10 # simply take 10 pages as example
for i in range(start, end + 1):
    url = 'http://konachan.net/post?page=%d&tags=' % i
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')
    for img in soup.find_all('img', class_="preview"):
        target_url = 'http:' + img['src']
        filename = os.path.join('imgs', target_url.split('/')[-1])
        download(target_url, filename)
    print('%d / %d' % (i, end))
```


    
    1 / 10
  
    
    2 / 10

    
    3 / 10
    
    
    4 / 10
   
    
    5 / 10
  
    
    6 / 10
    
    
    7 / 10 
    
    8 / 10
   
    
    9 / 10
    
    10 / 10


### extract faces


```python
import cv2
import sys
import os.path
from glob import glob

def detect(filename, cascade_file="lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(48, 48))
    for i, (x, y, w, h) in enumerate(faces):
        face = image[y: y + h, x:x + w, :]
        face = cv2.resize(face, (96, 96))
        save_filename = '%s-%d.jpg' % (os.path.basename(filename).split('.')[0], i)
        cv2.imwrite("faces/" + save_filename, face)
```


```python
if os.path.exists('faces') is False:
    os.makedirs('faces')
file_list = glob('imgs/*.jpg')
for filename in file_list:
    detect(filename)
```


```python
face_list = glob('faces/*.jpg')

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

plt.rcParams["figure.figsize"] = (20,20)
fig=plt.figure()
j=1
list=[]
while j<=25:
    i=np.random.randint(len(face_list))
    if i not in list:
        list.append(i)
    else:
        continue
    img=cv2.imread(face_list[i])
    fig.add_subplot(5,5,j)
    plt.imshow(img)
    plt.axis("off")
    j+=1
```


![png](output_23_0.png)

