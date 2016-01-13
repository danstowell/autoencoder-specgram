
# Spectrogram auto-encoder
# Dan Stowell 2016.
#
# Unusual things about this implementation:
#  * Data is not pre-whitened, instead we use a custom layer (NormalisationLayer) to normalise the mean-and-variance of the data for us. This is because I want the spectrogram to be normalised when it is input but not normalised when it is output.
#  * It's a convolutional net but only along the time axis; along the frequency axis it's fully-connected.

import numpy as np

import theano
import theano.tensor as T
import lasagne
#import downhill
from lasagne.nonlinearities import rectify, leaky_rectify, very_leaky_rectify
from numpy import float32

try:
	from lasagne.layers import InverseLayer as _
	use_maxpool = True
except ImportError:
	print("""**********************
		WARNING: InverseLayer not found in Lasagne. Please use a more recent version of Lasagne.
		WARNING: We'll deactivate the maxpooling part of the network (since we can't use InverseLayer to undo it)""")
	use_maxpool = False

import matplotlib
#matplotlib.use('PDF') # http://www.astrobetter.com/plotting-to-a-file-in-python/
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
plt.rcParams.update({'font.size': 6})

from userconfig import *
import util
from layers_custom import *

###################################################################################################################
# create Theano variables for input minibatch
input_var = T.tensor4('X')
# note that in general, the main data tensors will have these axes:
#   - minibatchsize
#   - numchannels (always 1 for us, since spectrograms)
#   - numfilts (or specbinnum for input)
#   - numtimebins

if example_is_audio:
	# load our example audio file as a specgram
	examplegram = util.standard_specgram((util.load_soundfile(examplewavpath, 0)))
	print("examplegram is of shape %s" % str(np.shape(examplegram)))

###################################################################################################################
# here we define our "semi-convolutional" autoencoder
# NOTE: lasagne assumes pooling is on the TRAILING axis of the tensor, so we always use time as the trailing axis

def make_custom_convlayer(network, in_num_chans, out_num_chans):
	"Applies our special padding and reshaping to do 1D convolution on 2D data"
	network = lasagne.layers.PadLayer(network, width=(featframe_len-1)/2, batch_ndim=3) # NOTE: the "batch_ndim" is used to stop batch dims being padded, but here ALSO to skip first data dim
	print("shape after pad layer: %s" % str(network.output_shape))
	network = lasagne.layers.Conv2DLayer(network, out_num_chans, (in_num_chans, featframe_len), stride=(1,1), pad=0, nonlinearity=very_leaky_rectify, W=lasagne.init.Orthogonal()) # we pad "manually" in order to do it in one dimension only
	filters = network.W
	network = lasagne.layers.ReshapeLayer(network, ([0], [2], [1], [3])) # reinterpret channels as rows
	print("shape after conv layer: %s" % str(network.output_shape))
	return network, filters

network = lasagne.layers.InputLayer((None, 1, specbinnum, numtimebins), input_var)
print("shape after input layer: %s" % str(network.output_shape))
#
# normalisation layer
#  -- note that we deliberately normalise the input but do not undo that at the output.
#  -- note that the normalisation params are not set by the training procedure, they need to be set before training begins.
network = NormalisationLayer(network, specbinnum)
normlayer = network # we need to remember this one so we can set its parameters
#
network, filters_enc = make_custom_convlayer(network, in_num_chans=specbinnum, out_num_chans=numfilters)
#
# NOTE: here we're using max-pooling, along the time axis only, and then
# using Lasagne's "InverseLayer" to undo the maxpooling in one-hot fashion.
# There's a side-effect of this: if you use *overlapping* maxpooling windows,
# the InverseLayer may behave slightly unexpectedly, adding some points with
# double magnitude. It's OK here since we're not overlapping the windows
if use_maxpool:
	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(1,2), stride=(1,2))
	maxpool_layer = network # store a pointer to this one

# NOTE: HERE is the "middle" of the autoencoder!
latents = network  # we remember the "latents" at the midpoint of the net, since we'll want to inspect them, and maybe regularise them too

if use_maxpool:
	network = lasagne.layers.InverseLayer(network, maxpool_layer)

network, filters_dec = make_custom_convlayer(network, in_num_chans=numfilters, out_num_chans=specbinnum)

network = lasagne.layers.NonlinearityLayer(network, nonlinearity=rectify)  # finally a standard rectify since nonneg (specgram) target

###################################################################################################################
# define simple L2 loss function with a mild touch of regularisation
prediction = lasagne.layers.get_output(network)
loss = lasagne.objectives.squared_error(prediction, input_var)
loss = loss.mean() + 1e-4 * lasagne.regularization.regularize_network_params(network, lasagne.regularization.l2)

###################################################################################################################

plot_probedata_data = None
def plot_probedata(outpostfix, plottitle=None):
	"""Visualises the network behaviour.
	NOTE: currently accesses globals. Should really be passed in the network, filters etc"""
	global plot_probedata_data

	if plottitle==None:
		plottitle = outpostfix

	if np.shape(plot_probedata_data)==():
		if example_is_audio:
			plot_probedata_data = np.array([[examplegram[:, examplegram_startindex:examplegram_startindex+numtimebins]]], float32)
		else:
			plot_probedata_data = np.zeros((minibatchsize, 1, specbinnum, numtimebins), dtype=float32)
			for _ in range(5):
				plot_probedata_data[:, :, np.random.randint(specbinnum), np.random.randint(numtimebins)] = 1

	test_prediction = lasagne.layers.get_output(network, deterministic=True)
	test_latents    = lasagne.layers.get_output(latents, deterministic=True)
	predict_fn = theano.function([input_var], test_prediction)
	latents_fn = theano.function([input_var], test_latents)
	prediction = predict_fn(plot_probedata_data)
	latentsval = latents_fn(plot_probedata_data)
	if False:
		print("Probedata  has shape %s and meanabs %g" % ( plot_probedata_data.shape, np.mean(np.abs(plot_probedata_data ))))
		print("Latents has shape %s and meanabs %g" % (latentsval.shape, np.mean(np.abs(latentsval))))
		print("Prediction has shape %s and meanabs %g" % (prediction.shape, np.mean(np.abs(prediction))))
		print("Ratio %g" % (np.mean(np.abs(prediction)) / np.mean(np.abs(plot_probedata_data))))

	util.mkdir_p('pdf')
	pdf = PdfPages('pdf/autoenc_probe_%s.pdf' % outpostfix)
	plt.figure(frameon=False)
	#
	plt.subplot(3, 1, 1)
	plotdata = plot_probedata_data[0,0,:,:]
	plt.imshow(plotdata, origin='lower', interpolation='nearest', cmap='RdBu', aspect='auto', vmin=-np.max(np.abs(plotdata)), vmax=np.max(np.abs(plotdata)))
	plt.ylabel('Input')
	plt.title("%s" % (plottitle))
	#
	plt.subplot(3, 1, 2)
	plotdata = latentsval[0,0,:,:]
	plt.imshow(plotdata, origin='lower', interpolation='nearest', cmap='RdBu', aspect='auto', vmin=-np.max(np.abs(plotdata)), vmax=np.max(np.abs(plotdata)))
	plt.ylabel('Latents')
	#
	plt.subplot(3, 1, 3)
	plotdata = prediction[0,0,:,:]
	plt.imshow(plotdata, origin='lower', interpolation='nearest', cmap='RdBu', aspect='auto', vmin=-np.max(np.abs(plotdata)), vmax=np.max(np.abs(plotdata)))
	plt.ylabel('Output')
	#
	pdf.savefig()
	plt.close()
	##
	for filtvar, filtlbl, isenc in [
		(filters_enc, 'encoding', True),
		(filters_dec, 'decoding', False),
			]:
		plt.figure(frameon=False)
		vals = filtvar.get_value()
		#print("        %s filters have shape %s" % (filtlbl, vals.shape))
		vlim = np.max(np.abs(vals))
		for whichfilt in range(numfilters):
			plt.subplot(3, 8, whichfilt+1)
			# NOTE: for encoding/decoding filters, we grab the "slice" of interest from the tensor in different ways: different axes, and flipped.
			if isenc:
				plotdata = vals[numfilters-(1+whichfilt),0,::-1,::-1]
			else:
				plotdata = vals[:,0,whichfilt,:]

			plt.imshow(plotdata, origin='lower', interpolation='nearest', cmap='RdBu', aspect='auto', vmin=-vlim, vmax=vlim)
			plt.xticks([])
			if whichfilt==0:
				plt.title("%s filters (%s)" % (filtlbl, outpostfix))
			else:
				plt.yticks([])

		pdf.savefig()
		plt.close()
	##
	pdf.close()

plot_probedata('init')

###################################################################################################################
if True:
	###################################
	# here we set up some training data. this is ALL A BIT SIMPLE - for a proper experiment we'd prepare a full dataset, and it might be too big to be all in memory.
	training_data_size=100
	training_data = np.zeros((training_data_size, minibatchsize, 1, specbinnum, numtimebins), dtype=float32)
	if example_is_audio:
		# manually grab a load of random subsets of the training audio
		training_data_size=100
		for which_training_batch in range(training_data_size):
			for which_training_datum in range(minibatchsize):
				startindex = np.random.randint(examplegram.shape[1]-numtimebins)
				training_data[which_training_batch, which_training_datum, :, :, :] = examplegram[:, startindex:startindex+numtimebins]
	else:
		# make some simple (sparse) data that we can train with
		for which_training_batch in range(training_data_size):
			for which_training_datum in range(minibatchsize):
				for _ in range(5):
					training_data[which_training_batch, which_training_datum, :, np.random.randint(specbinnum), np.random.randint(numtimebins)] = 1

	###################################
	# pre-training setup

	# set the normalisation parameters manually as an estimate from the training data
	normlayer.set_normalisation(training_data)

	###################################
	# training

	# compile training function that updates parameters and returns training loss
	params = lasagne.layers.get_all_params(network, trainable=True)
	updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=0.01, momentum=0.9)
	train_fn = theano.function([input_var], loss, updates=updates)

	# train network
	numepochs = 2048 # 3 # 100 # 5000
	for epoch in range(numepochs):
		loss = 0
		for input_batch in training_data:
			loss += train_fn(input_batch)
		if epoch==0 or epoch==numepochs-1 or (2 ** int(np.log2(epoch)) == epoch):
			lossreadout = loss / len(training_data)
			infostring = "Epoch %d/%d: Loss %g" % (epoch, numepochs, lossreadout)
			print(infostring)
			plot_probedata('progress', plottitle="progress (%s)" % infostring)

	plot_probedata('trained', plottitle="trained (%d epochs; Loss %g)" % (numepochs, lossreadout))

