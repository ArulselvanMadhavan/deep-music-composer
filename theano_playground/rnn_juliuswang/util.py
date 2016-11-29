import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cnf_matrix):
    np.set_printoptions(precision=2)

    np.set_printoptions(precision=2)
    title = 'Normalized confusion matrix on MNIST test set'
    class_name = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(class_name))
    plt.xticks(tick_marks, class_name)
    plt.yticks(tick_marks, class_name)

    thresh = cnf_matrix.max() / 2.
    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, cnf_matrix[i, j],
                 horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True digit')
    plt.xlabel('Predicted digit')
    plt.show()

def load_training_log(filename='train.log'):
    with open(filename, 'rb') as infile:
        tr_acc = pickle.load(infile)
        val_acc = pickle.load(infile)
        test_acc = pickle.load(infile)
        conf_matrix = pickle.load(infile)
    #print tr_acc, val_acc, test_acc
    #print conf_matrix
    return tr_acc, val_acc, test_acc, conf_matrix
