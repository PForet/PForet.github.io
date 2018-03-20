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

header:
    teaser: /assets/images/dev_PCA.png
excerpt: "This post should display a **header with an overlay image**, if the theme supports it."
---

# Principal Components Analysis for image classification

In image recognition, we generally overlook other techniques that were used before neural networks became standard. These techniques are still worth our time, as they present some advantages:
 - They are usually simpler and faster to implement.
 - If the database is small, they can outperform deep-learning methods.
 - When your grandchildren will ask how the job was done before quantum deep reinforcement learning was a thing, you will have a great story to tell.

For these reasons, I often start addressing an image classification problem without neural networks if possible, in order to get an idea of the "minimum" performance I should get when switching to more powerful algorithms.
Always know the basics. You don't want to be the guy who does sentiment analysis using deep pyramid CNN and doesn't realise a naive bayes classifier gives better results on his 50MB dataset.

So today, we will see how to recognise hand-written characters using simple machine learning algorithms.

## Classifying some Devanagari characters

Devanagari is an alphabet used in India and Nepal, composed of 36 consonants and 12 vowels. We will also add to this the 10 digits used to write numbers. For this classification problem, we will use a [small database available on Kaggle](https://www.kaggle.com/ashokpant/devanagari-character-dataset), composed of approximately 200 images of each class, hand written by 40 distinct individuals.

Some characters from the dataset are displayed below. The ones on the upper row are all different types of characters, but some of them can be really similar to a novice eye (like mine). On the other hand, the lower row shows some ways to write the same consonant, _"chha"_.

![Some characters]({{ "/assets/images/devanagari/characters examples.png" | absolute_url }})

With 58 classes of 200 images each and such an intra-class diversity, this problem is non-trivial. Today, we will build a classifier for each dataset of characters (consonant, vowel of numeral) separately. We will see how to achieve an accuracy between 97% (for the numerals) and 75% (for the consonants), using only scikit learn's algorithms. In another article, we will see how deep learning can push these results up to 99.7% for the numerals and 94.9% for the consonants.

## Dealing with images

We start by loading the images we want to classify, using `PIL` (Python Image Library). A demonstration code for that can be found [here](https://github.com/PForet/Devanagari_recognition/blob/master/load_data.py) if needed, but let's assume we already have a list of PIL images, and a list of integers representing their labels:

{% highlight python %}
consonants_img, consonants_labels = load_data.PIL_list_data('consonants')
{% endhighlight %}

For the sake of exposition, we will display the code only for the consonants dataset. Just assume everything is the same for the two others.
At this point, you might want to rescale all your images to the same dimensions, if it is not already done. Luckily, images from this dataset are all 36 x 36 pixels (thanks to you, kind Kaggle stranger).

We convert our images to black and white, and take their negative. PIL allows us to do that easily:

{% highlight python %}
from PIL import ImageOps  

def pre_process(img_list):
    img_bw = [img.convert('LA') for img in img_list]
    return [ImageOps.invert(img) for img in img_list]

consonants_proc = pre_process(consonants_img)
{% endhighlight %}


Finally, we must transform our images into vectors. In order to accomplish that, we transform each image into a matrix representing the pixels activation (a zero for a black pixel, and a 255 for a white one). We then rescale each element of the matrix by dividing it by the maximum possible value (255), before flattening the matrix:

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

Cheers ! The tedious part of pre-processing the images is over now.

## Import sklearn

Or as I call it, the poor man's `import keras`. After just some a few lines of code and we will be done classifying our images. Once satisfied, we will try to understand what happened exactly.

### Choosing the best model
Here, we choose to use a support vector machine classifier (SVC) on the reduced features returned by a principal component analysis (PCA, we will get back to that later). The SVC is well adapted when we have few samples (these things quickly become painfully slow as the number of samples grows).

These classifiers have a lot of meta-parameters, but we will tune here only C and gamma. We choose to use a gaussian kernel, the default one which works usually very well. We thus define a simple function that takes a vector of inputs and a vector of labels as arguments, tests several sets of parameters, and returns the best SVC found. In order to do that, we use Scikit's `GridSearch` that will test all the combinations of parameters from a dictionary, compute an accuracy with a K-Fold, and return the best model.

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def best_SVC(X,y):
    # Initiate a SVC classifier with default parameters
    svc_model = SVC()
    # The values to test for the C and gamma parameters.
    param_dic = {'C':[1,10,100],
                'gamma':[0.001,0.005,0.01]}
    clf = GridSearchCV(svc_model, param_dic, n_jobs=-1)
    # Search for the best set of parameters for our dataset, using bruteforce
    clf.fit(X, y)
    print("Best parameters: ", clf.best_params_)
    # We return the best model found
    return clf.best_estimator_
```

### Splitting the dataset

As usual, we split our dataset into a training set to train the model on, and a testing set to evaluate its results. We use Scikit's `train_test_split` function that is straightforward, and keep the default training/testing ratio of 0.8/0.2. Don't use the testing set to tune C and gamma, that's cheating.

### Using a PCA
Our input space is large: we have a dimension for each pixel of the picture: we thus have 1296 features by observation. We choose to use a PCA to reduce this number of dimensions to 24. That's were the magic happens.

### Computing the result

Our pipeline is very simple: given a list of inputs and a list of labels:
 - We split the lists to obtain a training set and a testing set.
 - We find the axis that maximises variance on the training set (using `pca.fit`).
 - We project the training and testing points on these axes (using `pca.transform`).
 - We find the SVC model that maximises the accuracy and fit it on the training set.
 - We compute the accuracy of this model on the test set and return it

```python
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

def benchmark(X,y):

    X_train, X_test, y_train, y_test = train_test_split(X,y)
    pca = PCA(n_components = 24)
    pca.fit(X_train)
    reduced_X_train, reduced_X_test = pca.transform(X_train), pca.transform(X_test)

    best_model = best_SVC(reduced_X_train,y_train)
    predictions = best_model.predict(reduced_X_test)
    return accuracy_score(y_test, predictions)
```

And we run this function on our three sets:

```python
score_on_numerals = benchmark(numerals_inputs, numerals_labels)
print("Best accuracy on numerals: {}".format(score_on_numerals))

score_on_vowels = benchmark(vowels_inputs, vowels_labels)
print("Best accuracy on vowels: {}".format(score_on_vowels))

score_on_consonants = benchmark(consonants_inputs, consonants_labels)
print("Best accuracy on consonants: {}".format(score_on_consonants))
```

Which should give something along these lines:
 ```
 ('Best parameters: ', {'C': 10, 'gamma': 0.01})
Best accuracy on numerals: 0.972222222222
('Best parameters: ', {'C': 10, 'gamma': 0.01})
Best accuracy on vowels: 0.906485671192
('Best parameters: ', {'C': 10, 'gamma': 0.005})
Best accuracy on consonants: 0.745257452575
```

Here we got the promised 97% accuracy on the numerals. That was easy. Remember a good code is like a good dentist: quick, and without unnecessary agonising pain. But now that we made this work, maybe it's time to understand what this PCA thing did to our images...

## PCA, mon amour,

### The basic idea

From now, we have two ways of explaining things:
 - With linear algebra, spectral decomposition, singular values and covariance matrix.
 - With some pictures.

If you want to explore the mathematical side of this (and you should, as it is not so difficult and PCA is fundamental in statistics), you will find plenty of good resources online. I like [this one](http://www.cs.otago.ac.nz/cosc453/student_tutorials/principal_components.pdf) which is complete and introduces all the algebra tools needed.

However, if you still can find your inner child, you'll follow me though the picture book explanation!

Let's start with a small set of amazing pictures that could easily belong in a MoMA collection:

![Some characters]({{ "/assets/images/devanagari/demo_pca.png" | absolute_url }})

Those are 5 by 5 unicolour pictures, so they could be represented by a vector of 25 dimensions. However, this is a bit too much, as this images seems so have some repeated patterns...

![And their components]({{ "/assets/images/devanagari/pca_toy.png" | absolute_url }})

Indeed, all of the eight pictures can be represented by adding some of these patterns. That would definitely give an advantage for classifying them: we need only a vector of size 4 to encode these pictures, and not 25 as before. Of course, a vector of size 25 is easily dealt with by most machine learning algorithms, but remember we used the same technique to reduce the dimension of our Devanagari characters from 1296 to 24.
But the main advantage is not here. Remember that we flatten our image into a vector where each dimension represents a pixel. Considering each pixel as a dimension has obvious drawbacks: a translation of one pixel for a character will lead to a very different point in the input vector space and, by opposition to CNN, most general machine learning algorithms don't take into account the relative positions of the pixels.

That's how principal component analysis will help. It uncovers spacial patters in our images. In fact, the PCA will "group" together the pixels which are activated simultaneously in our images. Pixels which are close to one another will have good chances of been activated simultaneously: the pen will leave a mark on both of them! But let's see how it worked out for our hand written characters.

### And now the real world application

We start by displaying the patters uncovered by the PCA, for the three datasets:
![Nepali components]({{ "/assets/images/devanagari/pca_main.png" | absolute_url }})

Those patterns are ordered according to the variance they explain. In other word, if there is a pattern that is composed by a lot of pixels that are often activated simultaneously, we say it explains a lot of variance. On the other hand, a very small and uncommon pattern is most likely noise and isn't really useful.

For instance, if we take a look at the first row, we will see that the most important pattern is a "O" shape, meaning this pattern is often repeated in our images. If we feed the vector returned by the PCA to a machine learning algorithm, it will have access to the information "there is a big 'O' shape on the image" only by looking at the first element of this vector. That will surely be useful to learn how to classify "zero"!

But how many patterns should we keep in our vectors? One way to decide is to visualise how many are needed to get a good reconstitution of our original images:

![PCA recomposition]({{ "/assets/images/devanagari/pca_recomposition.png" | absolute_url }})

Here, we show the original numerals on the top row, and the reconstituted images using 1, 4, 8, 24 and 48 patterns. We observe that using 24 patterns, we get a pretty good reconstitution of the original. That's the number we will put in `PCA(n_components = 24)`. Another way to find this number would be with tries and errors (there is a pretty good range of correct values), or looking at the proportion of explained variance, if you have a good grasp on how the PCA works.

### That's pretty much it

I hope you now have some understanding of how the PCA can be applied to image classification. Please keep in mind that PCA is a really powerful tool that can tackle a lot of statistics problems. We only have scraped the surface, and occulted some important points (the fact that the patterns are uncorrelated, for instance). So if you are not totally familiar with this tool, don't hesitate to do some research and have some practice !
