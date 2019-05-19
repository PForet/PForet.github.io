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
    teaser: /assets/images/svc.png
    overlay_image: /assets/images/head/head5.png
excerpt: "How to specialize a convolutional neural network, by replacing the last layers with a SVC."
---

# Specializing a neural network with SVC

_This article is a follow up to [this post](/characters-recognition-with-keras/), where we trained a CNN to recognise Devanagari characters._

Transfer learning is the practice of using knowledge already acquired to perform new tasks, and it's awesome. Of course, why would you start from scratch when the problem is almost already solved?

In the case of neural networks, a way to perform transfer learning is to re-train the last layers of a network. I'm not fond of this method, as it can feel unnatural to implement. I think a more explicit way to benefit from a trained network is to use it as a features extractor by chopping off the last layers. When you see your network as just a features extractor, retraining the last layers mean stacking a new network on top. But why should we limit ourselves to this possibility? If we have few training samples, why not add a more suitable algorithm like support vector classifiers, for instance?

This is exactly what we are going to do in this final chapter of our series on handwritten characters recognition. Today, we will improve greatly our accuracy by training a CNN on the whole database (numerals, consonants, and vowels), before replacing the last layers with support vector machines.

_This article follows [this one](/characters-recognition-with-keras/) and presupposes the same data structures are loaded in the workspace_

### Merging the datasets

**We start by merging the three sets**. To do so, we increment the labels of the vowels by 10 (the number of numerals) and the labels of the consonants by 22 (number of numerals + vowels) to resolve conflicts between labels.

**We then split the dataset into a training, a validation and a testing set**, using the same method as before (stratifying, and using the same proportions).

{% highlight python %}
all_tensors = np.vstack((tensors_numerals, tensors_vowels, tensors_consonants))
all_labels_int = numerals_labels + [int(i)+10 for i in vowels_labels] + [int(i)+22 for i in consonants_labels]

nb_labels = len(set(all_labels_int))
all_labels = np_utils.to_categorical([i-1 for i in np.array(all_labels_int).astype(int)], nb_labels)
X_model, X_test, y_model, y_test = train_test_split(all_tensors, all_labels, test_size = 0.15, stratify=all_labels, random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_model, y_model, test_size = 0.176, stratify=y_model, random_state=1)
{% endhighlight %}

### Building a new model

**We then define a model that will be trained on the whole training dataset** (numerals, consonants, and vowels together). We now have a more consequent dataset (over 9000 images for training), and we will use data-augmentation. Because of that, **we can afford a more complex model to better fit the new diversity of our dataset**. The new model is constructed as followed:

- We start with a convolutional layer with **more filters (128)**.
- We put **two dense layers of 512 nodes** before the last layer, to construct a better representation of the features uncovered by the convolutional layers. We will keep these layers when specialising the model to one of the three datasets.
- Because the model is still quite simple (no more than 4 millions parameters), we can afford to perform **numerous epochs during the training on a GPU**. Numerous epochs are also a good way to benefit fully from data-augmentation, as the model will discover new images at each iteration. However, to prevent overfitting, **we put a drop out layer before each dense layer**, and also one after the first convolutional layers.

The `keras` implementation of the model is:

{% highlight python %}
model_for_all = Sequential()
model_for_all.add(Conv2D(128, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(36,36,1)))
model_for_all.add(Dropout(0.25))
model_for_all.add(Conv2D(64, (3, 3), activation='relu'))
model_for_all.add(MaxPooling2D(pool_size=(2, 2)))
model_for_all.add(Conv2D(32, (3,3), activation='relu'))
model_for_all.add(Dropout(0.25))
model_for_all.add(Flatten())
model_for_all.add(Dense(512, activation='relu'))
model_for_all.add(Dropout(0.7))
model_for_all.add(Dense(512, activation='relu'))
model_for_all.add(Dropout(0.7))
model_for_all.add(Dense(58, activation='softmax'))

model_for_all.summary()
opt = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model_for_all.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
{% endhighlight %}

### Fitting the model with data-augmentation

**We will now fit the same model using data-augmentation**. We use Keras' `ImageDataGenerator` to dynamically generate new batches of images. We specify the transformations we want on augmented images:
- A **small random rotation** of the characters (maximum 15 degrees)
- A **small random zoom** (in or out), up to a maximum of 20% of the image size.
- We could add random translations, but they are pretty useless if the first layers are convolutional.

When using data-augmentation, we need to fit the model using a special function, `fit_generator`. We specify that **we want to monitor the training with a non-augmented validation set**, by specifying `validation_data=(X_val, y_val)`. Finally, we save the weights only when the validation loss is decreasing, and we predict the accuracy on the testing set.

{% highlight python %}
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.2)

# The checkpointer allows us to monitor the validation loss and to save weights
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.for_all',
                           verbose=1, save_best_only=True)

model_for_all.fit_generator(datagen.flow(X_train, y_train, batch_size=400),
                        steps_per_epoch=len(X_train) / 400, epochs=150,
                            validation_data=(X_val, y_val), callbacks=[checkpointer])

#The last weights are not the ones we want to keep, so we must reload the best weights found
model_for_all.load_weights('saved_models/weights.best.for_all')

y_pred = model_for_all.predict(X_test)
print("Accuracy on test set: {}".format(
        accuracy_score(np.argmax(y_test,axis=1), np.argmax(y_pred,axis=1))))
{% endhighlight %}

**We now remove the last two layers of our model** (the last dense layer and the drop out layer before). We then freeze the remaining layers to make them non-trainable, and we save our base model for future use.

{% highlight python %}
# Removes the two last layers (dense and dropout)
model_for_all.pop(); model_for_all.pop()
# Makes the layers non trainable
for l in model_for_all.layers:
    l.trainable = False
# Saves the model for easy loading
model_for_all.save("Models/model_for_all")
{% endhighlight %}

We should now split our whole dataset into its numerals, vowels and consonants components. This part is a little tedious but is necessary to ensure that we will test and validate our specified model on samples that were not seen during the previous learning.

To do that, we define an `extract_subset` function that allows us to extract samples for which the label is in a given range. For instance, to extract only the consonants, we should extract all the samples with a label between 0 and 9.

{% highlight python %}
def extract_subset(X_set, y_set, start_index, end_index):
    X = [[x] for (x,i) in zip(X_set,y_set) if np.argmax(i) >= start_index and np.argmax(i) < end_index]
    y = [[i[start_index:end_index]] for i in y_set if np.argmax(i) >= start_index and np.argmax(i) < end_index]
    return np.vstack(X), np.vstack(y)

# numerals : labels between 0 and 9
X_test_numerals, y_test_numerals = extract_subset(X_test, y_test, 0, 10)
# vowels : labels between 10 and 21
X_test_vowels, y_test_vowels = extract_subset(X_test, y_test, 10, 22)
# Consonants : labels between 22 and 58
X_test_consonants, y_test_consonants = extract_subset(X_test, y_test, 22, 58)

X_val_numerals, y_val_numerals = extract_subset(X_val, y_val, 0, 10)
X_val_vowels, y_val_vowels = extract_subset(X_val, y_val, 10, 22)
X_val_consonants, y_val_consonants = extract_subset(X_val, y_val, 22, 58)

X_train_numerals, y_train_numerals = extract_subset(X_train, y_train, 0, 10)
X_train_vowels, y_train_vowels = extract_subset(X_train, y_train, 10, 22)
X_train_consonants, y_train_consonants = extract_subset(X_train, y_train, 22, 58)
{% endhighlight %}

We have extracted the training, validation and testing sets for the three datasets. We can now load the pre-trained model, using keras' `load_model` function, and extract the activation of the last layers. These activations can be seen as high-level features of our images.

One way to specialize our model would be to add another dense layer (with a softmax activation) on the top. This is, in fact, equivalent to **performing a logistic regression over the activation of the last layers** produced by each image (these activations will be called 'bottleneck_features'). Thus, we suggest here to use a more powerful classifier instead of a last dense layer. We will train a SVC to predict the class of the character, given the bottleneck features as input.

Of course, our features extractor will be the model trained on the whole dataset (using data-augmentation) with the last dense layer removed. It will then transform an image into a vector of 512 high-level features that we can feed to our SVC.

{% highlight python %}
from keras.models import load_model
features_extractor = load_model("Models/model_for_all")
{% endhighlight %}

We then merge the training and validation set (to perform a K-fold for validation instead), and perform a grid search to find the best parameters for our SVC.
The following function will do so and return the best SVC found during the grid search.

{% highlight python %}
def extract_bottleneck_features(X):
    # Features are the activations of the last layer
    return features_extractor.predict(X)

def SVC_Top(X, y):
    # Scikit SVC takes an array of integers as labels
    y_flat = [np.argmax(i) for i in y]
    # We train the SVC on the extracted features
    X_flat = extract_bottleneck_features(X)

    High_level_classifier = SVC()
    # The parameters to try with grid_search
    param_grid = {'C':[1,10,100,1000],
                  'gamma':[0.001,0.01,0.1,1,10,100]}

    grid_search = GridSearchCV(High_level_classifier, param_grid)
    grid_search.fit(X_flat, y_flat)
    # We return the best SVC found
    return grid_search.best_estimator_

# We get the best SVC for each type of characters
X_consonants = np.vstack((X_train_consonants, X_val_consonants))
Best_top_classifier_consonants = SVC_Top(X_consonants, np.vstack((y_train_consonants, y_val_consonants)))

X_vowels = np.vstack((X_train_vowels, X_val_vowels))
Best_top_classifier_vowels = SVC_Top(X_vowels, np.vstack((y_train_vowels, y_val_vowels)))

X_numerals = np.vstack((X_train_numerals, X_val_numerals))
Best_top_classifier_numerals = SVC_Top(X_numerals, np.vstack((y_train_numerals, y_val_numerals)))
{% endhighlight %}

### Results

These results improved by far what we obtained with a CNN trained from scratch:

|            |SVC over bottleneck features | CNN from scratch | SVC with PCA |
|------------|:----------------------------:|:----------------:|:-------------:|
| Numerals   |             99.7%            |       97.9%      |        96.9% |
| Vowels     |             99.5%            |       93.5%      |        87.9% |
| Consonants |             94.9%            |      85.9%       |        75.0% |

On the vowels and the numerals, we achieve an accuracy of 99.5% and 99.7%, thus improving by 2.2 and 0.6 points the accuracy of the previous CNN model. The results are even more spectacular with the consonants, where we improve our accuracy from 85.9% up to 94.9% (+9.0 points).

## Neural networks as features extractors

The fact that a trained neural network can be used as a features extractor is very useful. For image recognition, a popular technique consists in using pre-trained CNN (such as [Inception or VGG](https://towardsdatascience.com/neural-network-architectures-156e5bad51ba)) to extract high-level features that can be fed to other machine learning algorithms. By doing so, I save myself the struggle of training a CNN and improve my precision by using the fact that these CNN were trained on a database far bigger than mine.

To visualize this phenomenon, I trained another CNN, with a dense layer containing only two neurons somewhere in the middle. If we remove all the layers after this one, the output of this CNN will be a vector of size two representing the input image in the plane. Please note that the final accuracy of this CNN is far inferior: it is generally a bad idea to put such a bottleneck on the information flowing in a neural network.

The features discovered by this CNN are displayed below (one colour by class, logarithmic transformations applied):

![Features 2d]({{ "/assets/images/devanagari/Features2d_all.jpeg" | absolute_url }})

Or, for clarity, if we keep only the vowels:

![Features 2d vowels]({{ "/assets/images/devanagari/Features2d_vowels.jpeg" | absolute_url }})

Here, we can see that the features extracted from the images are grouped by classes: this is why it is so efficient to use them as inputs to another classifier.

## Conclusion

We tested several ways to classify Devanagari characters. The first one was a Support Vector Classifier trained over the first 24 axes of a PCA. We then improved our accuracy by switching to a Convolutional Neural Network, trained only on the relevant dataset (consonants, vowels or numerals). At last, we trained another CNN on all the dataset, using data-augmentation, to provide a powerful features extractor. We then trained one specialized SVC for each type of character over the high levels features provided by this CNN. **With this technique, we achieved an accuracy far superior to the other methods (99.7% for the numerals, 99.5% for the vowels and 94.9% for the consonants.**

That's the end of the series! Thank for your attention, and I promise no more Devanagari characters here ;)
