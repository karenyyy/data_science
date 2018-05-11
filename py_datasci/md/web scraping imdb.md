

```python
from selenium import webdriver
from bs4 import BeautifulSoup

PATH=input("Enter url: ")
SAVE_PATH=input("Enter save path: ")
driver=webdriver.Chrome()
driver.get(PATH)
html_src=driver.page_source
soup=BeautifulSoup(html_src, 'lxml')

with open(SAVE_PATH, 'w') as file:
    file.write(soup.prettify())
driver.quit()
```

    Enter url: https://indico.io/blog/visualizing-with-t-sne/
    Enter save path: /home/karen/Desktop/test.html


Find by id:


```python
html_doc = """
<html>
<head>
<title>The Dormouse's story</title>
</head>
<body>
<p class="title">
<b>The Dormouse's story</b>
</p>
<p class="story">Once upon a time there were three little sisters; and their names were
<a href="http://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="http://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="http://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>
<p class="story">...</p>
</body>
</html>
"""

soup = BeautifulSoup(html_doc, 'lxml')

a = soup.find_all('a', {'id':'link1'})

print(a)
```

    [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]


Find by text:


```python
a_elsie = soup.find_all('a', string = 'Elsie')

print(a_elsie)
```

    [<a class="sister" href="http://example.com/elsie" id="link1">Elsie</a>]


Find siblings:


```python
first_a = soup.find('a') # find -> return the first found
remain_sibling = first_a.findNextSiblings()
print(remain_sibling)
```

    [<a class="sister" href="http://example.com/lacie" id="link2">Lacie</a>, <a class="sister" href="http://example.com/tillie" id="link3">Tillie</a>]


Search in scope:


```python
first_p = soup.find('p')
first_p
```




    <p class="title">
    <b>The Dormouse's story</b>
    </p>




```python
print(first_p.find('a'))
```

    None



```python
print(first_p.find('b'))
```

    <b>The Dormouse's story</b>


Search text:


```python
print(first_p.text)
```

    
    The Dormouse's story
    


Scrape links:


```python
a_tags = soup.find_all('a')

for a in a_tags:
    print(a['href'])
```

    http://example.com/elsie
    http://example.com/lacie
    http://example.com/tillie


Scrape data inside the tables:


```python
for tr in soup.find_all('tr'):
    for td in tr.find_all('td'):
        print(td.text)
```

### Example: IMDB dataset scraping


```python
from bs4 import BeautifulSoup
from selenium import webdriver

class Film():

    def __init__(self):
        self.rank = ""
        self.title = ""
        self.year = ""
        self.link = ""


def get_film_list():

    driver = webdriver.Chrome()

    url = 'http://www.imdb.com/chart/top?ref_=nv_mv_250_6'

    driver.get(url)

    soup = BeautifulSoup(driver.page_source,'lxml')

    table = soup.find('table', class_ = 'chart')

    film_list = []

    for td in table.find_all('td', class_ = 'titleColumn'):
        full_title = td.text.strip().replace('\n','').replace('      ','')
        rank = full_title.split('.')[0]
        title = full_title.split('.')[1].split('(')[0]
        year = full_title.split('(')[1][:-1]
        a = td.find('a')

        new_film = Film()
        new_film.rank = rank
        new_film.title = title
        new_film.year = year
        new_film.link = a['href']

        film_list.append(new_film)

    driver.quit()

    return film_list


film_list = get_film_list()
i=0
for f in film_list:
    i+=1
    if i==10:
        break
    print(f.title)
    print(f.rank)
    print(f.year)
    print(f.link)
    
```

    The Shawshank Redemption
    1
    1994
    /title/tt0111161/?pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=e31d89dd-322d-4646-8962-327b42fe94b1&pf_rd_r=0SX8A3SS7EWV4FTY4KZG&pf_rd_s=center-1&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_tt_1
    The Godfather
    2
    1972
    /title/tt0068646/?pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=e31d89dd-322d-4646-8962-327b42fe94b1&pf_rd_r=0SX8A3SS7EWV4FTY4KZG&pf_rd_s=center-1&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_tt_2
    The Godfather: Part II
    3
    1974
    /title/tt0071562/?pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=e31d89dd-322d-4646-8962-327b42fe94b1&pf_rd_r=0SX8A3SS7EWV4FTY4KZG&pf_rd_s=center-1&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_tt_3
    The Dark Knight
    4
    2008
    /title/tt0468569/?pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=e31d89dd-322d-4646-8962-327b42fe94b1&pf_rd_r=0SX8A3SS7EWV4FTY4KZG&pf_rd_s=center-1&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_tt_4
    12 Angry Men
    5
    1957
    /title/tt0050083/?pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=e31d89dd-322d-4646-8962-327b42fe94b1&pf_rd_r=0SX8A3SS7EWV4FTY4KZG&pf_rd_s=center-1&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_tt_5
    Schindler's List
    6
    1993
    /title/tt0108052/?pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=e31d89dd-322d-4646-8962-327b42fe94b1&pf_rd_r=0SX8A3SS7EWV4FTY4KZG&pf_rd_s=center-1&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_tt_6
    Pulp Fiction
    7
    1994
    /title/tt0110912/?pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=e31d89dd-322d-4646-8962-327b42fe94b1&pf_rd_r=0SX8A3SS7EWV4FTY4KZG&pf_rd_s=center-1&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_tt_7
    The Lord of the Rings: The Return of the King
    8
    2003
    /title/tt0167260/?pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=e31d89dd-322d-4646-8962-327b42fe94b1&pf_rd_r=0SX8A3SS7EWV4FTY4KZG&pf_rd_s=center-1&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_tt_8
    The Good, the Bad and the Ugly
    9
    1966
    /title/tt0060196/?pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=e31d89dd-322d-4646-8962-327b42fe94b1&pf_rd_r=0SX8A3SS7EWV4FTY4KZG&pf_rd_s=center-1&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_tt_9


Scraping and get posters


```python
import requests

url = 'http://www.imdb.com/title/tt0111161/?pf_rd_m=A2FGELUUNOQJNL&pf_rd_p=2398042102&pf_rd_r=0PS12P50E86XYMR1RVR3&pf_rd_s=center-1&pf_rd_t=15506&pf_rd_i=top&ref_=chttp_tt_1'

driver = webdriver.Chrome()

driver.get(url)

soup = BeautifulSoup(driver.page_source,'lxml')

div = soup.find('div', class_ = 'poster')

a = div.find('a')

print('http://www.imdb.com' + a['href'])

url = 'http://www.imdb.com' + a['href']

driver.get(url)

soup = BeautifulSoup(driver.page_source, 'lxml')

all_div = soup.find_all('div', class_ = 'pswp__zoom-wrap')

all_img = all_div[1].find_all('img')

print(all_img[1]['src'])

f = open('first_image.jpg', 'wb')
f.write(requests.get(all_img[1]['src']).content)
f.close()

driver.quit()
```

    http://www.imdb.com/title/tt0111161/mediaviewer/rm10105600?ref_=tt_ov_i
    https://images-na.ssl-images-amazon.com/images/M/MV5BMDFkYTc0MGEtZmNhMC00ZDIzLWFmNTEtODM1ZmRlYWMwMWFmXkEyXkFqcGdeQXVyMTMxODk2OTU@._V1_.jpg



```python
# div class_=poster --> a['href'] --> div[1] class_=pswp__zoom-wrap --> img[1]['src']

from selenium import webdriver
from bs4 import BeautifulSoup
import requests

class Film():
    """docstring for Film"""
    def __init__(self):
        self.rank = ""
        self.title = ""
        self.year = ""
        self.link = ""


def get_film_list():

    driver = webdriver.Chrome()

    url = 'http://www.imdb.com/chart/top?ref_=nv_mv_250_6'

    driver.get(url)


    soup = BeautifulSoup(driver.page_source,'lxml')

    table = soup.find('table', class_ = 'chart')

    film_list = []

    for td in table.find_all('td', class_ = 'titleColumn'):

        full_title = td.text.strip().replace('\n','').replace('      ','')
        rank = full_title.split('.')[0]
        title = full_title.split('.')[1].split('(')[0]
        year = full_title.split('(')[1][:-1]
        a = td.find('a')

        new_film = Film()
        new_film.rank = rank
        new_film.title = title
        new_film.year = year
        new_film.link = a['href']

        film_list.append(new_film)

    driver.quit()

    return film_list


def download_all_posters(film_list):

    driver = webdriver.Chrome()

    for film in film_list:
        url = 'http://www.imdb.com/' + film.link
        driver.get(url)
        soup = BeautifulSoup(driver.page_source,'lxml')

        div = soup.find('div', class_ = 'poster')

        a = div.find('a')
        
        url = 'http://www.imdb.com' + a['href']

        driver.get(url)

        soup = BeautifulSoup(driver.page_source, 'lxml')

        all_div = soup.find_all('div', class_ = 'pswp__zoom-wrap')

        all_img = all_div[1].find_all('img')

        f = open('../imdb/{0}.jpg'.format(film.title.replace(':','')), 'wb')
        f.write(requests.get(all_img[1]['src']).content)
        f.close()

    driver.quit()

```


```python
download_all_posters( get_film_list() )
```


```python
def creat_driver():
    driver = webdriver.Chrome()
    return driver


def get_download_link_one_page(url):

    link_list_one_page = []

    driver = create_driver()

    driver.get(url)

    soup = BeautifulSoup(driver.page_source, 'lxml')

    div = soup.find('div', class_ = 'media_index_thumb_list')

    for a in div.find_all('a'):
        link_list_one_page.append(a['href'])

    driver.quit()

    return link_list_one_page

def get_download_link_all_pages():
    max_page = 1
    link_list_all_page = []

    for page_number in range(1, max_page + 1):
        page_url = 'http://www.imdb.com/gallery/rg1528338944?page={0}&ref_=rgmi_mi_sm'.format(page_number)
        link_list_all_page.extend(get_download_link_one_page(page_url))

    return link_list_all_page


def download_latest_poster(link_list_all_page):

    driver = create_driver()
    for link in link_list_all_page:
        driver.get('http://www.imdb.com'+link)
        soup = BeautifulSoup(driver.page_source,'lxml')
        divs = soup.find_all('div', class_= 'pswp__zoom-wrap')
        imgs = divs[1].find_all('img')
        jpg_link = imgs[1]['src']
        
        # get image title
        div = soup.find('div', class_ = 'mediaviewer__footer')
        p = div.find('p')
        jpg_title = p.text
        
        f = open('../imdb2/{0}.jpg'.format(jpg_title.replace(':','')),'wb')
        f.write(requests.get(jpg_link).content)
        f.close()

    driver.quit()


download_lastest_poster(get_download_link_all_pages())
```
