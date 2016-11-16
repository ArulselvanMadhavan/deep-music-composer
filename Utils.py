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