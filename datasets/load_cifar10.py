import numpy as np
import pickle
import os


def unpickle(file):
    """
        Unpickled a CIFAR 10 zip file
    """
    assert os.path.exists(file), "[ERROR]: Data file not found!"
    
    with open(file, "rb") as fo:
        data = pickle.load(fo, encoding="bytes")
    return data


def load_cifar10(data_dir, negatives=False):
    assert os.path.exists(data_dir), "[ERROR] Dataset directory not found!"
    
    meta_data_dict = unpickle(os.path.join(data_dir, "batches.meta"))
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)
    
    # Load training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []
    
    for i in range(1, 6):
        cifar_train_data_dict = unpickle(os.path.join(data_dir, "data_batch_{}".format(i)))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']
        
    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    
    
    # Load test data
    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)
    
    print("[INFO] Finished loading the CIFAR-10 dataset, datastats: ")
    print("[INFO] Train images shape: {}".format(cifar_train_data.shape))
    print("[INFO] Train labels shape: {}".format(len(cifar_train_labels)))
    print("[INFO] Test images shape: {}".format(cifar_test_data.shape))
    print("[INFO] Test labels shape: {}".format(len(cifar_test_labels)))
    
    return cifar_train_data, cifar_train_filenames, cifar_train_labels, cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names
    