Clone from:
https://github.com/IraKorshunova/folk-rnn


To train:

source activate folkrnn

 THEANO_FLAGS='floatX=float32,device=cuda0,gpuarray.preallocate=1' python train_rnn.py config5 data/allabcwrepeats_parsed
