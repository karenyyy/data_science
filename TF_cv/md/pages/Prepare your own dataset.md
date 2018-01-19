
# Prepare my own Dataset

[download data here](https://www.kaggle.com/c/dogs-vs-cats/data)


```python
import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
import re
import shutil
import os
%matplotlib inline

tf_keras = tf.contrib.keras
```

## Separate Cat and Dog Images


```python
def get_cat_and_dog_files(data):

    cat_files = []
    dog_files = []

    for string in data:
        # file example: '../data/train/cat.10005.jpg'
        result = re.search("\/([a-z]+)\.", string)

        if result is not None:
            if result.group(1) == "cat":
                cat_files.append(string)
            elif result.group(1) == "dog":
                dog_files.append(string)
        else:
            print("ERROR: can't find a match for: {}".format(string))
    return cat_files, dog_files       
```

# Dataset size


```python
dataset_path = "data/train/*.jpg"
data = glob.glob(dataset_path)

print("{} images found:".format(len(data)))

cat_files, dog_files = get_cat_and_dog_files(data)

print("{} cat images".format(len(cat_files)))
print("{} dog images".format(len(dog_files)))

```

    25000 images found:
    12500 cat images
    12500 dog images


# Visualize Dataset


```python
def plot_image_grid(images_files):
    # figure size
    fig = plt.figure(figsize=(8, 8))
    
    # load images
    images = [tf_keras.preprocessing.image.load_img(img) for img in images_files]
    
    # plot image grid
    for x in range(4):
        for y in range(4):
            ax = fig.add_subplot(4, 4, 4*y+x+1)
            plt.imshow(images[4*y+x])
            plt.xticks(np.array([]))
            plt.yticks(np.array([]))
    plt.show()
```

# Cat Sample Images


```python
plot_image_grid(cat_files[:16])
```


![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/output_9_0.png)


# Dog Sample Images


```python
plot_image_grid(dog_files[:16])
```


![png](https://raw.githubusercontent.com/karenyyy/data_science/master/TF_cv/images/output_11_0.png)


# Split Dataset Into Train and Test Set


```python
def train_test_split(cat_files, dog_files):
    train_cat_files = cat_files[:10000]
    test_cat_files = cat_files[10000:]

    train_dog_files = dog_files[:10000]
    test_dog_files = dog_files[10000:]

    print("train size: {} cats and {} dogs".format(len(train_cat_files), len(train_dog_files)))
    print("test size :  {} cats and  {} dogs".format(len(test_cat_files), len(test_dog_files)))
    
    return train_cat_files, test_cat_files, train_dog_files, test_dog_files
```


```python
train_test_files = train_test_split(cat_files, dog_files)
```

    train size: 10000 cats and 10000 dogs
    test size :  2500 cats and  2500 dogs


# Save Train Test Split to New Directory


```python
# Create new directory and copy files to it
def copy_files_to_directory(files, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print("Created directory: {}".format(directory))

    for f in files:
        shutil.copy(f, directory)

# combine all the dataset preparation steps into one function
def prepare_cat_dog_dataset(dataset_path):
    train_cat_dir = "data/training/cat/"
    train_dog_dir = "data/training/dog/"
    test_cat_dir = "data/testing/cat/"
    test_dog_dir = "data/testing/dog/"

    print("Load images...")
    data = glob.glob(dataset_path)
    print("{} images found".format(len(data)))

    print("\nSeperate cat images from dog images")
    cat_files, dog_files = get_cat_and_dog_files(data)

    print("\nSplit train and test dataset")
    train_cat_files, test_cat_files, train_dog_files, test_dog_files = train_test_split(cat_files, dog_files)

    print("\nCopying train cat images to new directory...")
    copy_files_to_directory(train_cat_files, train_cat_dir)

    print("\nCopying test cat images to new directory...")
    copy_files_to_directory(test_cat_files, test_cat_dir)

    print("\nCopying train dog images to new directory...")
    copy_files_to_directory(train_dog_files, train_dog_dir)

    print("\nCopying test cat images to new directory...")
    copy_files_to_directory(test_dog_files, test_dog_dir)
```


```python
dataset_path = "data/train/*.jpg"
prepare_cat_dog_dataset(dataset_path)
```

    Load images...
    25000 images found
    
    Seperate cat images from dog images
    
    Split train and test dataset
    train size: 10000 cats and 10000 dogs
    test size :  2500 cats and  2500 dogs
    
    Copying train cat images to new directory...
    Created directory: data/training/cat/
    
    Copying test cat images to new directory...
    Created directory: data/testing/cat/
    
    Copying train dog images to new directory...
    Created directory: data/training/dog/
    
    Copying test cat images to new directory...
    Created directory: data/testing/dog/

