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
    teaser: /assets/images/devanagari-1-reduced.png
    overlay_image: /assets/images/head/head6.png
excerpt: "Performing characters recognition with Keras and TensorFlow"
toc: true
toc_label: "Contents"
toc_icon: "angle-double-down"
---

# Classifying hand written characters with Keras

In [this article](/PCA-for-image-classification/), we saw how to apply principal component analysis to image recognition. Our performances were quite good, but clearly not state-of-the art. Today, we are going to see how we can improve our accuracy using convolutional neural network (CNN). The best results will be obtained by combining CNN and support vector machines. This article is only meant as an introduction to CNN and `Keras`, so feel free to jump to the [last article of the serie](/neural-networks-specialization/) if you are already familiar with this framework.

## Simples CNN beats our PCA-SVC approach

As one could guess, a simple CNN is enough to improve the results obtained in the previous post. We explain here how to build a CNN using `Keras` (TensorFlow backend).

Several categories of neural networks are available on Keras, such as recurrent neural networks (RNN) or graph models. **We will only use sequential models**, which are constructed by stacking several neural layers.

### Choosing layers

We have several types of layers than we can stack in our model, including:

- **Dense Layers**: The simplest layers, where all the weights are independent and the layer is fully connected to the previous and following ones. These layers works well at the top of the network, analysing the high level features uncovered by the lower ones. However, they tend to add a lot of parameters to our model and make it longer to train.
- **Convolutional layers**: The layers from which the CNN takes its name. Convolutional layers work like small filters (with a size of often 3 or 4 pixels) that slide over the image (or the previous layer) and are activated when they find a special pattern (such as straight lines, of angles). Convolutional layers can be composed of numerous filters that will learn to uncover different patterns. They offer translation invariance to our model, which is very useful for image classification. In addition to this, they have a reasonable number of weights (usually much fewer than dense layers) and make the model faster to train compared to dense layers.
- **Pooling layers**: Pooling layers are useful when used with convolutional layers. They return the maximum activation of the neurons they take as input. Because of this, they allow us to easily reduce the output dimension of the convolutional layers.
- **Dropout layers**: These layers are very different from the previous ones, as they only serve for the training and not the final model. Dropout layers will randomly "disconnect" neurons from the previous layer during training. Doing so is an efficient regularisation technique that efficiently reduces overfitting (mode details below)

### Compiling the model

Once our model is built, we need to compile it before training. Compilation is done by specifying a loss, here the **categorical cross-entropy**, a metric (**accuracy** here) and an optimization method.
The loss is the objective function that the optimization method will minimize. Cross-entropy is a very popular choice for classification problems because it is differentiable, and reducing the cross-entropy leads to better accuracy. Choosing accuracy as our performance metric is fair only because our classes are well balanced in our datasets. I cannot emphasize enough how much accuracy would be a poor choice if our classes were imbalanced (more of some characters than others).

Finally, we use the **root mean square propagation (RMSprop)** as an optimization method. This method is a variant from the classic gradient descent method, which will adapt the learning rate for each weight. This optimizer allows us to tune the **learning rate** since, generally speaking, a smaller learning rate leads to better final results, even if the number of epochs needed for the training increase. Generally, this optimizer works well, and changing it has very minimal effects on performance.

With all these tools, we define a first model for the consonants dataset (just assume we do the same for the numerals and the vowels). This model is meant to be trained from scratch **without transfer learning or data-augmentation**, in order to allow us to quantify the improvements brought by these techniques in another article.

### A model to train from scratch

Now the fun part: we stack layers like pancakes, hoping we don't do something stupid. If you follow this basic reasoning, nothing should go wrong:

- We start with a **two-dimension convolutional layer**(because our images only have one channel, as we work with gray images). We specify the number of filters we want for this layer. 32 seems like a good compromise between complexity and performance. Putting **32 filters** in this layer means that this layer will be able to identify up to 32 different patterns. It is worth noting that raising this number to 64 doesn't improve the overall performance, but also doesn't make the model notably harder to train. We specify a kernel size: 3 pixels by 3 pixels seems like a correct size, as it is enough to uncover simple patterns like straight lines or angles, but not too big given the size of our inputs (only 36x36 pixels, the input shape). At last, we specify an activation function for this layer. We will use **rectified linear units (ReLU)**, as they efficiently tackle the issue of the [vanishing gradient](https://machinelearningmastery.com/exploding-gradients-in-neural-networks/).
- We then add another **convolutional layer**, to uncover more complicated patterns, this time with **64 filters** (as we expect that more complicated patters than simple patterns will emerge from our dataset). We keep the same kernel size and the same activation function.
- After that, we add a **max-pooling layer** to reduce the dimensionality of our inputs. The pooling layer has no weights or activation function, and will output the biggest value found in its kernel. We choose a **kernel size of 2 by 2**, to lose as little information as possible while reducing the dimension.
- After that pooling layer, we add a first **dense layer with 256 nodes** to analyze the patterns uncovered by the convolutional layers. Being fully connected to the previous layer and the following dense one, the size of this layer will have a huge impact on the total number of trainable parameters of our model. Because of that, we try to keep this layer reasonably small, while keeping it large enough to fit the complexity of our dataset. Because our images are not really complex, we choose a size of 256 nodes for this layer. We add a ReLU activation function, as we did in the previous layers.
- Finally, we add the **final dense layer**, with **one node for each class** (36 for the consonant dataset). Each node of this layer should output a probability for our image to belong to one of the classes. Therefore, we want our activation function to return values between 0 and 1, and thus choose a **softmax activation function** instead of a ReLU as before.

### Overfitting

Because of their complexity and their large number of weights, **neural networks are very prone** to overfitting. Overfitting can be observed when the accuracy on the training set is really high, but the accuracy on the validation set is much poorer. This phenomenon occurs when the model has learnt "by heart" the training observations but is no longer capable of generalizing its predictions to new observations. As a result, we should stop the training of our model when the accuracy on the validation set is no longer decreasing. Keras allows us to easily do that by saving the weights at each iteration, only if the validation score decreases.

However, if our model overfits too quickly, this method will stop the training too soon and the model will yield very poor results on the validation and testing sets. To counter that, we will use a **regularisation method**, preventing overfitting while allowing our model to perform enough iterations during the learning phase to be efficient.

The method we will use relies on **dropout layers**. Dropout layers are layers that will randomly "disconnect" neurons from the previous layer, meaning their activation for this training iteration will be null. By disconnecting different neurons randomly, we prevent the neural network to build overly specific structures that are only useful for learning the training observations and not the "concept" behind them.

To apply this method, **we insert two drop_out layers in our model**, before each dense layer. Drop_out layers require only one parameter: the probability of a neuron to be disconnected during a training iteration. These parameters should be adjusted with trials and errors, by monitoring the accuracy on the testing and validation set during training. We found that **25% for the first drop_out layer and 80% for the second** gives the best results.

### Implementation

We use `Keras` with a TensorFlow backend to implement our model:

{% highlight python %}
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import RMSprop

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(36,36,1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.8))
model.add(Dense(36, activation='softmax'))

model.summary()
opt = RMSprop(lr=0.0005, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
{% endhighlight %}

Also, we will implement a `get_score` function that will take as inputs the following:
- ***tensors***: A whole dataset as a tensor
- ***labels***: The corresponding labels
- ***model***: The untrained Keras model for which we want to compute the accuracy
- ***epoch***: An integer specifying the number of epochs for training
- ***batch_size***: An integer, the size of a batch for learning (the greater the better, if the memory allows it)
- ***name***: The name of the model (to save the weights)
- ***verbose***: An optional boolean (default is false) that determines if we should tell Keras to display information during the training (useful for experimentation).

The function will:
- **Perform one-hot encoding** on the labels, so they can be understood by the model.
- **Split our dataset** into a training, a validation and a testing sets as detailed above.
- Create a checkpointer which allows us to **save the weights during training** (only if the accuracy is still improving).
- **Fit the model on the training set** and **monitor its performances on the validation set** (to know when to save weights).
- Compute and print the accuracy on the testing set.
- Return the trained model with the best weights available.

{% highlight python %}

from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint  
from keras.utils import np_utils
from sklearn.metrics import accuracy_score

def get_score(tensors, labels, model, epoch, batch_size, name, verbose=False):

    nb_labels = len(set(labels)) #Get the number of disctinct labels in the dataset
    # Encode the labels (integers) into one-hot vectors
    y_all = np_utils.to_categorical([i-1 for i in np.array(labels).astype(int)], nb_labels)
    # Split the testing set from the whole set, with stratification
    X_model, X_test, y_model, y_test = train_test_split(tensors, y_all,
                                        test_size=0.15, stratify=y_all, random_state=1)
    # Then split the remaining set into a training and a validation set
    # We use a test size of 17.6% because our remaining set account for only 85% of the whole set
    X_train, X_val, y_train, y_val = train_test_split(X_model, y_model,
                                        test_size=0.176, stratify=y_model, random_state=1)
    # Display the sizes of the three sets
    print("Size of the training set: {}".format(len(X_train)))
    print("Size of the validation set: {}".format(len(X_val)))
    print("Size of the testing set: {}".format(len(X_test)))

    # Create a checkpointer to save the weights when the validation loss decreases
    checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.{}.hdf5'.format(name),
                               verbose=1, save_best_only=True)
    # Fit the model, using 'verbose'=1 if we specified 'verbose=True' when calling the function (0 else)
    model.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[checkpointer],
             batch_size=batch_size, epochs=epoch, verbose=(1 if verbose else 0))

    # Reload best weights before prediction, and predict
    model.load_weights('saved_models/weights.best.{}.hdf5'.format(name))
    y_pred = model.predict(X_test)

    # Compute and print the accuracy
    print("Accuracy on test set: {}".format(
            accuracy_score(np.argmax(y_test,axis=1), np.argmax(y_pred,axis=1))))

    return model # And return the trained model
{% endhighlight %}

We can now train our model and get our score:

{% highlight python %}
get_score(tensors_consonants, consonants_labels, model, epoch=180, batch_size=800,
          name='consonants_from_scratch')
{% endhighlight %}

By reporting the results obtained for the three datasets, we see improvements compared to the SVC methods.

|            | CNN from scratch | SVC with PCA |
|------------|:---------------:|:------------:|
| Numerals   |      97.9%      |        96.9% |
| Vowels     |      93.5%      |        87.9% |
| Consonants |      85.9%      |        75.0% |

Ultimately, it is possible to increase again this accuracy by training our models on bigger datasets. How can we have more training images with only the dataset we used? To answer this question, we will in the [last article of the series](/neural-networks-specialization/):
 - Use data-augmentation
 - Train a generic model on the three datasets, before specializing it by replacing the last layers by support vector machines.
