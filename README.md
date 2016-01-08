
Spectrogram auto-encoder
(c) Dan Stowell 2016.


A simple example of an autoencoder set up for spectrograms, with two convolutional layers - thought of as one "encoding" layer and one "decoding" layer.

It's meant to be a fairly minimal example of doing this in Theano, using the Lasagne framework to make things easier.

By default it simply makes a training set from different chunks of the same single spectrogram (from the supplied wave file). This is not a good training set!

Notable (potentially unusual) things about this implementation:
 * Data is not pre-whitened, instead we use a custom layer (NormalisationLayer) to normalise the mean-and-variance of the data for us. This is because I want the spectrogram to be normalised when it is input but not normalised when it is output.
 * It's a convolutional net but only along the time axis; along the frequency axis it's fully-connected.
 * There's no maxpooling/downsampling.


SYSTEM REQUIREMENTS
===================

* Python
* Theano (NOTE: please check the Lasagne page for preferred Theano version)
* Lasagne https://github.com/Lasagne/Lasagne
* Matplotlib
* scikits.audiolab

Tested on Ubuntu 14.04 with Python 2.7.

USAGE
=====

          python autoencoder-specgram.py

It creates a "pdf" folder and puts plots in there (multi-page PDFs) as it goes along.
There's a "progress" pdf which gets repeatedly overwritten - you should see the output quality gradually getting better.

Look in userconfig.py for configuration options.

