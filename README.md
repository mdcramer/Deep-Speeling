# Deep-Speeling
## Using Deep Learning to correct spelling mistakes
### Motivation
This project was inspired by [Tal Weiss'](https://medium.com/@majortal) post on [Deep Spelling](https://medium.com/@majortal/deep-spelling-9ffef96a24f6). His [Deep Spell](https://github.com/MajorTal/DeepSpell/blob/master/keras_spell.py) code can be found on Github.

In January 2017 I began the [Udacity Deep Learning Foundation Nanodegree Program](https://www.udacity.com/course/deep-learning-nanodegree-foundation--nd101) and was hooked from the first lecture. I'd heard the term 'neural network' plenty of times previously, and had a general idea of what they could accomplish, but never had an understanding of how they 'work.' Since completing the course I've hadn't had much opportunity to tinker with the technology, but I've continued to contemplate it's uses, particularly in the domain of information retreival, which is where I've spent the last decade focusing.

Unless you're Google, the typical technique for correcting spelling mistakes is the [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance), or it's close cousin, the [Damerau–Levenshtein distance](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance). Mr. Weiss does a good job of explaining why these do not work particularly well.

### Goals
* Re-implement Mr. Weiss' Recurrent Neural Network (RNN) using Tensorflow and achieve the same level of accuracy.
* Attempt to implement some of the areas for exploration he suggests, as well as others, to see if futher improvements can be obtained.

### The code
The first part of the code, which involves downloading the [billion word dataset](http://research.google.com/pubs/pub41880.html) that Google released and setting it up for training, is predominantly lifted from Mr. Weiss. The second part of the code, which involves building the graph and training the neural network, is borrowed from the Udacity [sequence-to-sequency RNN example](https://github.com/mdcramer/deep-learning/tree/master/seq2seq). This Udacity example works on a small dataset of 10,000 'sentences' (1- to 8-character words) and trains the network to sort the characters alphabetically. The current project contains code to handle both this large dataset as well as the small dataset, which is useful for debugging.

Several enhancements have been added to the code:
* [Dropout](https://en.wikipedia.org/wiki/Dropout_(neural_networks)) was added.
* The Udacity example saved the graph and then reloaded it to run a single example, but it has now been reworked to be saved and loaded for additional training.
* The script has been reworked so that it can be exported at a .py file and run top to bottom from the command line. A "small" command line switch was added to toggle running with the small dataset.
* The Validation section has been reworked to run on the entire validation set, with the accuracy being computed. It may now be running after each epoch.
* A 'send email' fuction that communicates the current state of training was created to send udpates while training on AWS.
* Random uniform initialization was replaced with [Gaussian initialization scaled by fan-in](https://www.tensorflow.org/versions/r0.12/api_docs/python/contrib.layers/initializers).

### Todo
After 24 hours of training on an EC2 g2.2xlarge the network was able to achieve 75% accuracy. This, however, is far short of the 90% accuracy achieved by Mr. Weiss in 12 hours of training. All of the major elements of his model have been implemented, which the exception of [Categorical Cross-Entropy](http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html#theano.tensor.nnet.nnet.categorical_crossentropy) for the loss, so this is the next order of business. Tensorflow doesn't seem to have the identical equivalent, so I'll be attempting [softmax cross entropy](https://www.tensorflow.org/api_docs/python/tf/nn/softmax_cross_entropy_with_logits). If this works I'll move on to trying to improve beyond Mr. Weiss. If not, well, I'll be stuck...
