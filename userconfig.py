
# Configuration options that you might like to change

example_is_audio = True   # if False, generates simple sparse data for probing; else loads an audio file
examplegram_startindex = 550   # just choosing which bit to plot

#examplewavpath = "~/birdsong/linhart2015mar/concatall/perfolder/PC1101-rep-day2.wav"
examplewavpath = "509.WAV"
examplewavpath = "renneschiffchaff20130320bout1filt.wav"

srate = 22050.
wavdownsample = 2  # eg 44 kHz audio, factor of 2, gets loaded as 22 kHz. for no downsampling, set this ratio to 1

audioframe_len    = 128
audioframe_stride = 64

specbinlow = 10
specbinnum = 32

featframe_len    = 9
featframe_stride = 16
numfilters       = 6
minibatchsize    = 16
numtimebins = 160 # 128 # 48 # NOTE that this size needs really to be compatible with downsampling (maxpooling) steps if you use them.


###########################################################
# Below, we calculate some other things based on the config

import os
examplewavpath = os.path.expanduser(examplewavpath)


hopsize_secs = audioframe_stride / float(srate)
print("Specgram frame hop size: %.3g ms" % (hopsize_secs * 1000))
specgramlen_secs = hopsize_secs * numtimebins
print("Specgram duration: %.3g s" % specgramlen_secs)

