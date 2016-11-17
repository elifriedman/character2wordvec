from make_dataset import Corpus
from keras.models import Model, Sequential
from keras.layers.core import Dense, Masking, Activation, Dropout
from keras.layers import LSTM 

def build_model(masking_value, Timesteps, Dimension):
    model = Sequential()
    model.add(Masking(mask_value = masking_value, input_shape=(Timesteps, Dimension)))
    model.add(LSTM(128))
    print "compiling..."
    model.compile("adam","mean_squared_error")
    print "done compiling"
    return model

import sys
def main(filenames):
    corpus = Corpus(filenames)
    model = build_model(corpus.masking_value, corpus.T, corpus.D)

    mat1 = corpus.onehotCorpus(0)
    pred = model.predict(mat1)
    print pred.shape, pred

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "usage: python " + sys.argv[0] + " <filename1> [<filenam2>, ...]"
        sys.exit(-1)
    main(sys.argv[1:])
