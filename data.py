import torch
import numpy as np
from torch.utils.data import Dataset

path = "/home/beast/DATA/DATASET_NPZ/"


def load_mnist(path=path+"MNIST_Combined.npz"):
    f = np.load(path)
    x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f[
        'x_test'], f['y_test']
    f.close()
    x = np.concatenate((x_train, x_test)).astype(np.float32)
    y = np.concatenate((y_train, y_test)).astype(np.int32)

    # x = np.expand_dims(x, axis=1).astype(np.float32)
    # x = np.divide(x, 255.)
    print('MNIST samples', x.shape)
    return x, y

def load_qmnist(path=path+"QMNIST_Combined.npz"):
    f = np.load(path)
    print(path)
    x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f[
        'x_test'], f['y_test']
    f.close()
    x = np.concatenate((x_train, x_test)).astype(np.float32)
    y = np.concatenate((y_train, y_test)).astype(np.int32)

    # x = np.expand_dims(x, axis=1).astype(np.float32)
    # x = np.divide(x, 255.)
    print('QMNIST samples', x.shape)
    return x, y


def load_emnist(path=path+"EMNIST_Combined.npz"):
    f = np.load(path)
    x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f[
        'x_test'], f['y_test']
    f.close()
    x = np.concatenate((x_train, x_test)).astype(np.float32)
    y = np.concatenate((y_train, y_test)).astype(np.int32)

    # x = np.expand_dims(x, axis=1).astype(np.float32)
    # x = np.divide(x, 255.)
    print('EMNIST samples', x.shape)
    return x, y


def load_fmnist(path="/home/tejas/experimentations/Myidec/IDEC-pytorch/data/fmnist.npz"):
    f = np.load(path)

    x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f[
        'x_test'], f['y_test']
    f.close()
    x = np.concatenate((x_train, x_test)).astype(np.float32)
    y = np.concatenate((y_train, y_test)).astype(np.int32)

    x = np.expand_dims(x, axis=1).astype(np.float32)

    print('FMNIST samples', x.shape)
    return x, y


def load_cifar10(path=path+"CIFAR10_Combined.npz"):
    f = np.load(path)

    x_train, y_train, x_test, y_test = f['x_train'], f['y_train'], f[
        'x_test'], f['y_test']
    f.close()
    x = np.concatenate((x_train, x_test)).astype(np.float32)
    y = np.concatenate((y_train, y_test)).astype(np.int32)

    print('cifar10 samples', x.shape)
    return x, y


class NPZ_Dataset(Dataset):

    def __init__(self, str="MNIST"):

        if str == "MNIST":
            self.x, self.y = load_mnist()

        if str == "QMNIST":
            self.x, self.y = load_qmnist()

        if str == "FMNIST":
            self.x, self.y = load_fmnist()

        if str == "CIFAR10":
            self.x, self.y = load_cifar10()

        if str == "EMNIST":
            self.x, self.y = load_emnist()


    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))
