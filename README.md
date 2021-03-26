### Neural Networks as Feature extractors

### Brief introduction into neural networks
Neural networks are some of the most powerful machine learning programs. There are many types, but the general architecture involves three key components. 
- An `input` layer, 
- followed by one or more `hidden` layers 
- and finally, an `output` layer.

<img src='https://www.ritchievink.com/img/post-9-mlp/nn_diagram_1.png'>

The hidden layer is where the magic happens. Each of the layers in a traditional neural network is comprised of many `neurons`. Each connected to every other neuron in both the previous and subsequent layer. 

A single neuron can expect to receive the input of every other neuron before it, multiplied by their respective `weights`. After collecting all these values, it would then apply an `activation` function, which is simply a non-linear function, on the sum. This new value would then be passed on to the next layer.
- Why is an activation function needed? Well because if it were not used, the neurons would be computing linear equations. There is nothing wrong with that, but we already have algorithms to apply linear models in traditional machine learning. Neural networks were introduced precisely to address the non-linearity in some data structure relationships.

The way neural networks are designed was inspired by how the brain is thought to work. And although they do show good results, it is not quite clear why exactly neural networks perform so well.

---
### Theory on why neural networks work
One theory consistent with the brain analogy is the idea of `manifold unwinding`<sup>1</sup>.

Even though an object can vary in many ways (perspective, lighting, distance etc.), people still have an easy time in classification. 

The insight from this? The identity of an image can encompass many variations, but there must be a way to identify uniqueness. 

For example, suppose we were to plot a face with axis representing each feature it contains. A person’s face can have many different variations. But it does not vary in every possible way, or else it would be undistinguishable from anyone else. Thus, there must be some region in this feature space that contains every possible version of the face, what we call a `manifold`.


Unfortunately, if we were to plot these feature spaces for data collected in the real world, it would look like manifolds `hopelessly entangled` together. Thus, one could reason that an algorithm that can do a good job at image classification is `untangling` these manifolds into distinct regions.  

<img src='https://computervisionblog.files.wordpress.com/2013/09/untangleinvariantobjectrecognition2.png'>

---

### Neural Networks for Feature Extraction

This idea of manifold untangling gives room for an interesting idea. If we were to conclude that the hidden layers are in fact doing some feature extraction under the hood. Can we use the outputs of these hidden layers to identify the features in our data? 

It would be interesting if we could feed these raw features into a traditional machine learning classifier. A research paper done by `Stephen Notley, Malik Magdon-Ismail`<sup>2</sup> explores exactly this idea. There experiments involved measuring the difference in accuracy of `neural networks`, `traditional machine learning classifiers`, and a model where the features of a neural network were fed to traditional machine learning classifiers. 

Unsurprisingly, there was not a large difference between the performance of neural networks alone and in combination with classifiers. I say this because if we were to believe that neural networks were untangling manifolds, then most of the important work would have been done in the hidden layers. That is to say, the _`"preprocessing"`_ performed by the hidden layers is more important than the complexity of the classifier object atteached at the end.

However, a difference was still noted, and in many of the tests, a model that incorporated both neural networks and some traditional classifier did a little better than the neural network alone. Which suggest that it may be reasonable to employ such strategies in certain circumstances. 

---
### Let's attempt to show this with some simple code
<br>

[Code being used](https://gist.github.com/yemaney/60e5a758d84a4c6cd52c4a4c02f81d86)

You can follow the code above to follow along. A high-level explanation for the steps taken in this example.

---
- load the data
- preprocess the data
- create a cnn model called `feature_extractor`
- create a final cnn model which takes the output from `feature_extractor`
and put its through a softmax layer for classification.
- create a random forest classifier model that takes the output from `feature_extractor` as an input for classification
- compare the accuracies of both methods 
---


### The data used here the standard cifar10 dataset that contains images of 10 different objects

```
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()
```

### Here we'll do some preproccesing to make the data appropraite for the models

```
X_train = X_train / 255
X_test = X_test / 255

def reduce_data(X_train, X_test, y_train, y_test, sample_size):
    return X_train[:sample_size], X_test[:sample_size], y_train[:sample_size], y_test[:sample_size]

X_train, X_test, y_train, y_test = reduce_data(X_train, X_test, y_train, y_test, 10000)   
```


### Modeling 

Now we still have three more  steps
1. create a CNN neural network, without a softmax output layer called `feature_extractor`
2. create a final CNN model by connect this feature extractor to a softmax output layer
    - train and evaluate
3. feature_extractor to creat outputs, to be fed into a random forest classifer
    - train and evaluate

### Feature extractor

```
feature_extractor = models.Sequential([
    # cnn
    layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPool2D((2,2)),
    #dense
    layers.Flatten(input_shape=(32,32,3)),
    layers.Dense(1000, activation='relu'),
])
```

### Main CNN model

``` 
#Add layers for deep learning prediction
x = feature_extractor.output  
x = Dense(128, activation = 'sigmoid', kernel_initializer = 'he_uniform')(x)
prediction_layer = Dense(10, activation = 'softmax')(x)

# Make a new model combining both feature extractor and x
cnn_model = Model(inputs=feature_extractor.input, outputs=prediction_layer)
cnn_model.compile(optimizer='adam',loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
print(cnn_model.summary()) 
```

### Train the CNN

``` 
#Train the CNN model
history = cnn_model.fit(X_train, y_train, epochs=10, validation_data = (X_test, y_test))
```
### Random forest 

```
RF_model = RandomForestClassifier(n_estimators = 50, random_state = 42)

#Now, let us use features from convolutional network for RF
X_for_RF = feature_extractor.predict(X_train)
#This is out X input to RF
#Send test data through same feature extractor process
X_test_feature = feature_extractor.predict(X_test)
```

```
# Train the model on training data
RF_model.fit(X_for_RF, y_train) #For sklearn no one hot encoding
```

### Make predictions on test sets with both models

```
prediction_RF = RF_model.predict(X_test_feature)

prediction_cnn = cnn_model.predict(X_test)
predicted_cnn = prediction_cnn.argmax(axis=1)
```

### Evaluation

```
#Print overall accuracy
from sklearn import metrics

print('### Printing accuracies of both models ###')
print ("Accuracy off CNN model = ", metrics.accuracy_score(y_test, predicted_cnn))
print ("Accuracy off CNN + RF model = ", metrics.accuracy_score(y_test, prediction_RF))
```

### If you followed correctly, you should get an output similar to below.

```
### Printing accuracies of both models ###
Accuracy off CNN model =  0.5719
Accuracy off CNN + RF model =  0.5726
```

---
Notice the model that used a random forest classifier on the data preprocessed by a neural network did a little better that the neural network on its own.

Although I must confess, I sort of used a cheat code to get these answers. If you remember, during the preprocessing steps, I decided to limit the amount of data I used for training my models. This was an effort to penalize the neural network, as their learning potential is correlated with the amount of data given.

Although a little contrived, this example shows that there may cases when a combination of neural networks and traditional machine learning classifiers may give.

<br>

---

`sources`
1. [Untangling invariant object
recognition](http://www.rowland.harvard.edu/rjf/cox/pdfs/TICS_DiCarloCox_2007.pdf)
2. [Examining the Use of Neural Networks for Feature Extraction](https://arxiv.org/abs/1805.02294)
