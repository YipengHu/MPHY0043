
import os

import tensorflow as tf

## dataset - the "motion class"
DIR_URL = "https://github.com/YipengHu/MPHY0030/blob/main/tutorials/statistical_motion_model/data/"
file_paths = {"FILE_TRAIN" : "nodes_train.npy", "FILE_TEST" : "nodes_test.npy", "FILE_TRIS" : "tris.npy"}

for f,fn in file_paths.items():
    dir_tmp = tf.keras.utils.get_file(fn, DIR_URL+fn, cache_dir=os.path.abspath('./'))
    file_paths[f] = dir_tmp
    print(dir_tmp)
