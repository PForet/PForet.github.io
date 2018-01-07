---
defaults:
  # _posts
  - scope:
      path: ""
      type: posts
    values:
      layout: single
      author_profile: true
      read_time: true
      comments: true
      share: true
      related: true
      use_math: true
---

# Principal Components Analysis for image classification

In image recognition, we generally overlook other techniques that were used before neural networks became a standard. These techniques are still worth our time, as they present some advantages:
 - They are usually simpler and faster to implement.
 - The process leading to the classification is better understood that for deep neural networks.
 - If the database is small, they can outperform deep-learning methods.
 
For these reasons, I often start to adress an image classification problem without neural networks if possible, to get an idea of the "minimum" performance I should get when switching to more powerful algorithms. 
 
So today, we will see how to proceed to hand-written characters recognition using simple machine learning algorithms.

## Classifying some Devanagari characters

Devanagari is an alphabet used in India and Nepal, composed of 36 consonants and 12 vowels. We will also add to this the 10 digits used to write numbers. For this classification problem, we will use a [small database available on Kaggle](https://www.kaggle.com/ashokpant/devanagari-character-dataset), composed of approximately 200 images of each class, hand written by 40 distinct individuals. 

Some characters from the dataset are displayed below. The upper row are all different type of characters, but some of them can be really similar in a novice eye (like mine). On the other hand, the lower row shows some ways to write the same consonant, _"chha"_. 

![Some characters]({{ "/assets/images/devanagari/characters examples.png" | absolute_url }})

With 58 classes of 200 images each and such an intra-class diversity, this problem is non-trivial. Today, we will build a classifier for each dataset of character (consonant, vowel of numeral) separatedly. We will see how to achieve an accuracy between 97% (for the numerals) and 75% (for the consonants), using only scikit learn's algorithms. In another article, we will see how deep learning can push these results up to 99.7% for the numerals and 94.9% for the consonants.

## Dealing with images

We start by loading the images we want to classify, using PIL (Python Image Library). A demontration code for that can be found [here](https://github.com/PForet/Devanagari_recognition/blob/master/load_data.py) if needed, but let's assume we already have a list of PIL images, and a list of integers representing their labels:

{% highlight python %}
consonants_img, consonants_labels = load_data.PIL_list_data('consonants')
{% endhighlight %}

For the sake of exposition, we will display the code only for the consonants dataset. Just assume everythong is the same for the two others.
At his point, you might want to rescale all your images to the same dimension, if it is not already done. Luckily, images from this dataset are all 36 x 36 pixels. 

We convert our images to black and white, and take their negative. PIL allows us to do that easily:

{% highlight python %}
from PIL import ImageOps  

def pre_process(img_list):
    img_bw = [img.convert('LA') for img in img_list]
    return [ImageOps.invert(img) for img in img_list]

consonants_proc = pre_process(consonants_img)
{% endhighlight %}


At last, we must transform our images in vectors. For that, we transform each image into a matrix representing the pixels activation (a zero for a black pixel, and a 255 for a white one). We rescale each element of the matrix by dividing it by the maximum possible value (255), before flattening the matrix:

{% highlight python %}
import numpy as np
def vectorize_one_img(img):
        # Represent the image as a matrix of pixel weights, and flatten it
        flattened_img = np.asmatrix(img).flatten()
        # Rescaling by dividing by the maximum possible value of a pixel
        flattened_img = np.divide(flattened_img,255.0)
        return np.asarray(flattened_img)[0]
{% endhighlight %}

And we apply this transformation to all the images of our dataset:

```python
def to_vectors(img_list):
    return [vectorize_one_img(img) for img in img_list]

consonants_inputs = to_vectors(consonants_proc)
```




