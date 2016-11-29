import os
import gzip
import numpy as np
import cPickle as pickle

def onehot(y, num_classes=10):
	onehot_vector = np.zeros((y.shape[0], num_classes)).astype(np.float32)
	onehot_vector[np.arange(y.shape[0]), y] = 1.0
	return onehot_vector

# MNIST dataset download
def load_data(dataset):
    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('... loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

    # Change MNIST pictures to a have the shape [batch, channels, height, width]

    train_set = (
        train_set[0].reshape((-1, 28, 28)),
        train_set[1]
    )
    valid_set = (
        valid_set[0].reshape((-1, 28, 28)),
        valid_set[1]
    )
    test_set = (
        test_set[0].reshape((-1, 28, 28)),
        test_set[1]
    )

    return train_set, valid_set, test_set
