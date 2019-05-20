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
    teaser: /assets/images/imdb.png
    overlay_image: /assets/images/head/head4.png
excerpt: "How to classify texts using Bayesian features and support vector machines."
toc: true
toc_label: "Contents"
toc_icon: "angle-double-down"
---

# IMDB sentiment analysis with Bayesian SVC.

Movies are great! Sometimes... But what if we want to find out if one is worth watching? A good start would be to look at its rating on the biggest reviewing platform, IMDB. We could also do without the rating, just by reading the reviews of other film enthusiasts, but this takes some time... so what about making our computer read the reviews and assess if they are rather positive or negative?

Thanks to the size of this database, this toy problem has been studied a lot, with different algorithms. [Aditya Timmaraju and Vikesh Khanna](https://cs224d.stanford.edu/reports/TimmarajuAditya.pdf) from Stanford University give a really nice overview of the various methods that can be used to tackle this problem, achieving a maximum accuracy of 86.5% with support vector machines. [James Hong and Michael Fang](https://cs224d.stanford.edu/reports/HongJames.pdf) used paragraph vectors and recurrent neural networks to classify correctly 94.5% of the reviews. Today, we explore a much simple algorithm, yet very effective, proposed by [Sida Wang and Christopher D. Manning](https://www.aclweb.org/anthology/P12-2018): the Naive Bayes Support Vector Machine (NBSVM). We will propose a geometric interpretation of this method, in addition to a Python implementation that yields **91.6%** of accuracy on the IMDB dataset in only a few lines of code.

## Multinomial Naive Bayes classifier

Bayesian classifiers are a very popular and efficient way to tackle text classification problems. With this method, we represent a text by a vector $$f$$ of occurrences, for which each element denotes the number of times a certain word appears in this text. The order of the words in the sentence doesn't matter, only the number of times each word appears. The Bayes formula gives us the probability that a certain text is a positive review (label $$Y=1$$):

$$
P(Y=1|f) = \frac{P(f|Y=1)P(Y=1)}{P(f)}
$$

We want to find the probability that a given text $$f$$ is a positive review ($$Y=1$$). Thanks to this formula, we only need to know the probability that this review, knowing that it is positive, was written. ($$P(f|Y=1)$$), and the overall probability that a review is positive $$P(Y=1)$$.
Although $$P(f)$$ appears in the formula, it does not really matter for our classification, as we will see.

$$P(Y=1)$$ can be easily estimated: it is the frequency of positive reviews in our corpus (noted $$\frac{N^+}{N}$$).
However, $$P(f|Y=1)$$ is more difficult to estimate, and we need to make some very strong assumptions about it.
In fact, we will consider that the appearance of each word of the text is independent of the appearance of the other words. This assumption is very _naive_, thus illustrating the name of the method.

We now consider that
$$f|Y$$
follows a multinomial distribution: for a review of $$n$$ words,
what is the probability that these words are distributed as in $$f$$?
If we denote
$$p_i$$
the probability that a given word
 $$i$$
appears in a positive review (and $$q_i$$ that it appears in a negative review), the multinomial distributions assume that $$f|Y$$ is distributed as follows:

$$
\begin{array}{lll}
P(f|Y=1) = \frac{(\sum_{i=1}^n f_i)!}{\prod_{i=1}^n f_i!}\prod_{i=1}^n p_i^{f_i} & \mbox{ and } &
P(f|Y=0) = \frac{(\sum_{i=1}^n f_i)!}{\prod_{i=1}^n f_i!}\prod_{i=1}^n q_i^{f_i}
\end{array}
$$

Thus, we can predict that the review is positive if
$$P(Y=1|f) \geq P(Y=0|f)$$
, that is if the likelihood ratio
$$L$$
is greater than one:

$$
L = \frac{P(Y=1|f)}{P(Y=0|f)} = \frac{P(f|Y=1)P(Y=1)}{P(f|Y=0)P(Y=0)} = \frac{\prod_{i=1}^n p_i^{f_i} \times \frac{N^+}{N}}{\prod_{i=1}^n q_i^{f_i} \times \frac{N^-}{N}}
$$

Or, equivalently, if its logarithm is greater than zero:

$$\ln(L) = \sum_{i=1}^n f_i \ln\left(\frac{p_i}{q_i}\right) + \ln\left(\frac{N^+}{N^-}\right)$$

Which can be written as:

$$\begin{array}{llll} w^T . f + b > 0 & \mbox{ with } & w = \ln\left(\frac{P}{Q}\right) & b = \ln\left(\frac{N^+}{N^-}\right) \end{array}$$

We see that our decision boundary is linear in the log-space of the features. However, I like to see this formula as written differently:

$$
1^T . (w\circ f) + b > 0
$$

where $$\circ$$ stands for the element-wise product and $$1$$ for the unitary vector $$(1,1,...1)$$.
Now our Bayesian features vector is $$(w\circ f)$$ and our hyperplane is orthogonal to $$1$$. However we can wonder if this particular hyperplane is the most efficient for classifying the reviews... and the answer is no! Here is our free lunch: we will use support vector machines to find a better separating hyperplane for these Bayesian features.

## From reviews to vectors

The original dataset can be found [here](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz). However, this script named [IMDB.py](https://github.com/PForet/ML-hack/blob/master/bayesian-features/IMDB.py) loads the reviews as a list of strings for both the train and the test sets:

{% highlight python %}
from IMDB import load_reviews

# Load the training and testing sets
train_set, y_train = load_reviews("train")
test_set, y_test = load_reviews("test")

{% endhighlight %}
Feel free to use it, it downloads and unzips the database automatically if needed. We will use `Scikit.TfidfVectorizer` to transform our texts into vectors. Instead of only counting the words, it will return their frequency and apply some very useful transformations, such as giving more weight to uncommon words. The vectorizer I used is a slightly modified version of `TfidfVectorizer` which a custom pre-processor and tokenizer (which keeps exclamation marks, useful for sentiment analysis). By default, it doesn't only count words but also bi-grams (pairs of consecutive words), as this gives best results at the cost of an increasing features space. You can find the code [here](https://github.com/PForet/ML-hack/blob/master/bayesian-features/text_processing.py), and use it to run your own test:

{% highlight python %}
from text_processing import string_to_vec
# Returns a vector that counts the occurrences of each n-gram
my_vectorizer = string_to_vec(train_set, method="Count")
# Returns a vector of the frequency of each n-gram
my_vectorizer = string_to_vec(train_set, method="TF")
# Same but applies an inverse document frequency transformation
my_vectorizer = string_to_vec(train_set, method="TFIDF")
{% endhighlight %}

You can tune every parameter of it, just as with a standard `TfidfVectorizer`. For instance, if you want to keep only individual words and not bi-grams:

{% highlight python %}
# Returns a vector that counts the occurrences of each word
my_vectorizer = string_to_vec(X_train, method="Count", ngram_range = (1))
{% endhighlight %}

From now on, we will only use:
{% highlight python %}
myvectorizer = string_to_vec(train_set, method="TFIDF")
{% endhighlight %}
This will keep all words and bi-grams that appear more than 5 times in our corpus. This is a lot of words: our features space has 133572 dimensions, for 25000 training points!
Now that we know how to transform our reviews to vectors, we need to choose a machine learning algorithm. We talked about support vector machines. However, they scale very poorly and are too slow to be trained on 25000 points with more than 100000 features. We will thus use a slightly modified version, the dual formulation of a _l2-penalized_ logistic regression. We will now explain why this is very similar to a support vector classifier.

## Support vector machine and logistic regression

### Cost function of a support vector machine

A support vector machine tries to find a separation plane $$w^T.f=b$$ that maximises the distance between the plane and the closest points. This distance, called _margin_, can be expressed in terms of $$w$$ :
{:refdef: style="text-align: center;"}
![support vector machine margin]({{ "/assets/images/stat/SVM_margin.png" | absolute_url }}){:height="50%" width="50%"}
{: refdef}
A point is correctly classified if it is on the good side of the plane, and outside of the margin. On this image, we see that a sample is correctly classified if  $$w^T.f + b > 1$$ and $$Y=1$$ or $$w^T.f + b < 1$$ and $$Y=0$$. This can be summarised as $$(2y_i - 1)(w^T.f+b)$$. We want to maximise the margin $$\frac{2}{||w||} > 1$$ thus the optimisation problem of a support vector classifier is:

$$
\left\{ \begin{array}{ll}
\min_{w,b} \frac{1}{2}||w||^2 \\
s.t. (2y_i - 1)(w^T.f+b) \geq 1
\end{array} \right.
$$

However, if our observations are not linearly separable, such a solution doesn't exist. Therefore we introduce _slack variables_ that allow our model to incorrectly classify some points at some cost $$C$$:

$$
\left\{ \begin{array}{ll}
\min_{w,b} \frac{1}{2}||w||^2 + C \sum_{i=1}^n \epsilon_i \\
s.t. (2y_i - 1)(w^T.f+b) \geq 1 - \epsilon_i \mbox{ with }
\epsilon_i \geq 0
\end{array} \right.
$$

### Logistic regression

In logistic regression, the probability of a label to be $$1$$ given a vector $$f$$ is:

$$
\begin{array}{lll}
P(Y=1 | \, f) = \sigma(w^T.f+b) & \mbox{ where } & \sigma(x) = \frac{1}{1+e^{-x}}
\end{array}
$$

If we add a _l2-regularisation_ penalty to our regression, the objective function becomes:

$$
\min_{w,b} \frac{1}{2}||w||^2 + C \sum_{i=1}^n \ln\left(1+e^{-y_i(w^T.f+b)}\right)
$$

Where $$\sum_{i=1}^n \ln\left(1+e^{-y_i(w^T.f+b)}\right)$$ is the negative log-likelihood of our observations.
If you like statistics, it is worth noting that adding the _l2-penalty_ is the same as maximising the likelihood with a Gaussian prior on the weights (or a Laplacian prior for a _l1-penalty_).

### Why are they similar?

We define the likelihood ratio as

$$
r = \frac{P(Y=1|\, f)}{P(Y=0| \,f)} = e^{w^Tf+b}
$$

the cost of a positive example for the support vector machine is:

$$
cost_{Y=1} = C \times max(0, 1-(w^T.f+b)) = C \times max(0, 1-log(r))
$$

and for the logistic regression with a _l2-regularisation_ penalty:

$$
cost_{Y=1} = C \times \ln\left(1+e^{-(w^T.f+b)}\right) = C \times \ln\left(1+\frac{1}{R}\right)
$$

If we plot the cost of a positive example for the two models, we see that we have very similar losses:
{:refdef: style="text-align: center;"}
![support vector machine against logit loss]({{ "/assets/images/stat/svc_vs_lr.png" | absolute_url }})
{: refdef}

This is why a SVC with a linear kernel will give results similar to a _l2-penalized_ linear regression.

In our classification problem, we have 25000 training examples, and more than 130000 features, so a SVC will be extremely long to train.
However, a linear classifier with a l2 penalty is much faster than a SVC when the number of samples grows, and gives very similar results, as we just saw.

### Dual formulation of the logistic regression

When the number of samples is fewer than the number of features, as it is here, one might consider solving the dual formulation of the logistic regression.
If you are interested in finding out about this formulation, I recommend [Hsiang-Fu Yu, Fang-Lan Huang, and Chih-Jen Lin](https://link.springer.com/content/pdf/10.1007%2Fs10994-010-5221-8.pdf) which makes a nice comparison between the linear SVC and the dual formulation of the logistic regression, uncovering more similarities between these techniques.

## Implementation of the model

As seen before, we define

$$ \begin{array}{lll} P = \alpha + \sum_{i, y_i=1} f_i & \mbox{ and } &
Q = \alpha + \sum_{i, y_i=0} f_i \end{array}$$

For some smoothing parameter $$\alpha$$. The log-ratio $$R$$ is defined as:

$$R = \ln\left(\frac{P/||P||_1}{Q/||Q||_1}\right)$$

Where
$$||.||_1$$
 stands for the $$L^1$$ norm.

At last, the Bayesian features used to fit our SVC will be

$$R \circ X_{train}$$

Of course, we will use a sparse matrix to save memory (our vectors are mostly zeros).
Wrapped in some python code, this gives:

{% highlight python %}
from __future__ import division
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
import numpy as np

class NBSVM:

    def __init__(self, alpha=1, **kwargs):
        self.alpha = alpha
        # Keep additional keyword arguments to pass to the classifier
        self.kwargs = kwargs

    def fit(self, X, y):
        f_1 = csr_matrix(y).transpose()
        f_0 = csr_matrix(np.subtract(1,y)).transpose() #Invert labels
        # Compute the probability vectors P and Q
        p_ = np.add(self.alpha, X.multiply(f_1).sum(axis=0))
        q_ = np.add(self.alpha, X.multiply(f_0).sum(axis=0))
        # Normalize the vectors
        p_normed = np.divide(p_, float(np.sum(p_)))
        q_normed = np.divide(q_, float(np.sum(q_)))
        # Compute the log-ratio vector R and keep for future uses
        self.r_ = np.log(np.divide(p_normed, q_normed))
        # Compute bayesian features for the train set
        f_bar = X.multiply(self.r_)
        # Fit the regressor
        self.lr_ = LogisticRegression(dual=True, **self.kwargs)
        self.lr_.fit(f_bar, y)

    def predict(self, X):
        return self.lr_.predict(X.multiply(self.r_))
{% endhighlight %}


And finally (I chose the parameters $$\alpha = 0.1$$ and $$C=12$$ with a cross-validation):
{% highlight python %}
# Transform the training and testing sets
X_train = myvectorizer.transform(train_set)
X_test = myvectorizer.transform(test_set)

clf = NBSVM(alpha=0.1,C=12)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
print("Accuracy: {}".format(accuracy_score(y_test, predictions)))
{% endhighlight %}

```
Accuracy: 0.91648
```
That was a pretty painless way of achieving 91.6% accuracy!

Thank you a lot for reading, and don't hesitate to leave a comment if you have any question or suggestion ;)
