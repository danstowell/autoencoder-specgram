
import numpy as np

import theano
import theano.tensor as T

from lasagne.layers.base import Layer

###############################################################################################################

class NormalisationLayer(Layer):
	"""
	This layer applies a simple mean-and-std normalisation to input data.
	This allows you to "learn" the mean+std from training data and then apply it "live" to any future incoming data.

	NOTE: the parameters are NOT learnt during training, but must be initialised BEFORE training using the set_normalisation() function.
	"""
	def __init__(self, incoming, numbins, **kwargs):
		"numbins is the number of frequency bins in the spectrograms we're going to be normalising"
		super(NormalisationLayer, self).__init__(incoming, **kwargs)
		self.numbins = numbins
		self._output_shape = None
		self.initialised = False
		# and for the normalisation, per frequency bin - typically, we "sub" the mean and then "mul" by 1/std (I write this as mul rather than div because often more efficient)
		self.normn_sub = theano.shared(np.zeros((1, 1, numbins,  1), dtype=theano.config.floatX), borrow=True, name='norm_sub', broadcastable=(1, 1, 0, 1))
		self.normn_mul = theano.shared(np.ones( (1, 1, numbins,  1), dtype=theano.config.floatX), borrow=True, name='norm_mul', broadcastable=(1, 1, 0, 1))
		# here we're defining a theano func that I can use to "manually" normalise some data if needed as a separate thing
		inputdata = T.tensor4('inputdata')
		self.transform_some_data = theano.function([inputdata], (inputdata - self.normn_sub) * self.normn_mul)

	def get_output_shape_for(self, input_shape):
		return input_shape

	def get_output_for(self, inputdata, **kwargs):
		#if not self.initialised:
		#	print("NormalisationLayer must be initalised with normalisation parameters before training")
		return (inputdata - self.normn_sub) * self.normn_mul

	def set_normalisation(self, databatches):
		numbins = self.numbins
		# we first collapse the data batches, essentially into one very long spectrogram...
		#print("databatches.shape: %s" % str(databatches.shape))
		data = np.concatenate(np.vstack(np.vstack(databatches)), axis=-1)
		#print("data.shape: %s" % str(data.shape))

		centre = np.mean(data, axis=1)
		self.normn_sub.set_value( centre.astype(theano.config.floatX).reshape((1,1,numbins,1)), borrow=True)
		self.normn_mul.set_value(1. / data.std( axis=1).reshape((1,1,-1,1)), borrow=True)

		self.initialised = True

