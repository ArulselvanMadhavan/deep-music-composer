"""
Author: Arulselvan Madhavan
github: https://github.com/Arulselvanmadhavan
"""

import numpy as np
import os
import glob
import os
import glob
import warnings
import matplotlib.pyplot as plt

class Utils(object):
    @staticmethod
    def validate_path(dir_path):
        if not os.path.exists(dir_path):
            raise Exception("Path:{} does not exist".format(dir_path))

    @staticmethod
    def get_matching_files(search_string):
        files = glob.glob(search_string)
        if(len(files) == 0):
            warnings.warn("No files found for {}".format(search_string))
        return files

    @staticmethod
    def visualize_stats(stats, filename, title):
        plt.subplot(2, 1, 1)
        plt.plot(stats['train_loss'], label="batch_loss")
        # plt.plot(stats["batch_accuracy"], label="batch_accuracy")
        plt.title('Loss history - '+title)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')

        plt.subplot(2, 1, 2)
        plt.plot(stats['train_acc_history'], label='train')
        # plt.plot(stats['val_acc_history'], label='val')
        plt.title('Classification accuracy history - '+title)
        plt.xlabel('Epoch')
        plt.ylabel('Clasification accuracy')
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()