import idx2numpy
import os


def load_mnist(data_dir):
    train_data_dir = os.path.join(data_dir, "train")
    test_data_dir = os.path.join(data_dir, "test")
    