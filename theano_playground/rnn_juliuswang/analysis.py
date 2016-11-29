from data_handler import load_data, onehot
from layer import layer_rnn
import numpy as np
import theano
import sys
import cPickle as pickle
import matplotlib.pyplot as plt
from util import load_training_log, plot_confusion_matrix

theano.config.floatX = 'float32'
theano.config.exception_verbosity = 'high'

def viz_U(preload_model, row_indices):

    with open(preload_model, 'rb') as infile:
        U = pickle.load(infile)
        V = pickle.load(infile)
        W = pickle.load(infile)

    n_rows = len(row_indices)
    fig = plt.figure()
    for i, row in enumerate(row_indices, 1):
        a = fig.add_subplot(n_rows, 1, i)
        a.set_title('row %d in U' % row)
        a.set_yticks([])
        U_row = U[:, row].reshape(1, U.shape[0])
        U_row = (U_row - np.min(U_row)) / (np.max(U_row) - np.min(U_row))
        plt.imshow(U_row, cmap='gray', interpolation='nearest')
    plt.tight_layout()
    plt.savefig('viz_U.png')
    plt.show()

result_log = 'experiment/train_step28.log'
model_file = 'model/rnn_step28.model'
# tr_acc, val_acc, test_acc are training/validation/test accuracy
# tr_acc and val_acc are of size (# of epochs, ), and test accuracy is just a scalar
# conf_matrix: confusion matrix on test data
tr_acc, val_acc, test_acc, conf_matrix = load_training_log(result_log)
print 'number of epochs:', len(tr_acc), '/ training acc. at each epoch:', tr_acc
print 'number of epochs:', len(val_acc), '/ validation acc. at each epoch:', val_acc
print 'test accuracy:', test_acc

# here plot the confusion matrix on test data
plot_confusion_matrix(conf_matrix)

# here visualize U matrix on specified rows from 0 to 199, e.g. [2, 3, 10, 50, 100]
# it has total 200 rows as there are 200 hidden units
viz_U(model_file, row_indices=[2, 3, 10, 50, 100])
