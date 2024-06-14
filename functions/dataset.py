import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functions.sub_functions import set_seed

class BinaryMnist(Dataset):
    def __init__(self, train=True, size=None, normalize=False, seed=0, label_noise=None):
        if normalize:
            mean = (0.1307,)
            std = (0.3081,)
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
        else:
            transform = transforms.ToTensor()
        self.mnist = torchvision.datasets.MNIST(root='./data', train=train,
                                                download=True, transform=transform)
        class1 = 1
        class2 = 7
        subset_indices = ((self.mnist.targets == class1) + (self.mnist.targets == class2)).nonzero(as_tuple=False).view(-1)
        if size:
            set_seed(seed)
            subset_indices = subset_indices[torch.randint(low=0, high=len(subset_indices), size=(size, ))]
        self.mnist.data = self.mnist.data[subset_indices]
        self.mnist.targets = self.mnist.targets[subset_indices]
        self.mnist.targets[self.mnist.targets == class1] = 1.
        self.mnist.targets[self.mnist.targets == class2] = -1.
        if label_noise:
            num_noise = int(len(self.mnist.targets) * label_noise)
            # labels = torch.tensor([1, -1])
            # p = torch.tensor([0.5, 0.5])
            # set_seed(seed)
            # idx = p.multinomial(num_samples=num_noise, replacement=True)
            random_labels = torch.randint(2, size=(num_noise,))
            random_labels[random_labels==0] = -1.
            self.mnist.targets[:num_noise] = random_labels

    def __getitem__(self, index):
        data, target = self.mnist[index]

        return index, data, target

    def __len__(self):
        return len(self.mnist)


class BinaryMnist_01(BinaryMnist):
    def __init__(self, train=True, size=None, normalize=False, seed=0, label_noise=None):
        super(BinaryMnist_01, self).__init__(train, size, normalize, seed, label_noise)
        self.mnist.targets[self.mnist.targets == -1.] = 0.


class FullMnist(Dataset):
    def __init__(self, train=True, normalize=False):
        if normalize:
            mean = (0.1307,)
            std = (0.3081,)
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
        else:
            transform = transforms.ToTensor()
        self.mnist = torchvision.datasets.MNIST(root='./data', train=train,
                                                download=True, transform=transform)

    def __getitem__(self, index):
        data, target = self.mnist[index]

        return index, data, target

    def __len__(self):
        return len(self.mnist)


class FullCifar10(Dataset):
    def __init__(self, train=True, normalize=False):
        if normalize:
            mean = (0.49139968, 0.48215827 ,0.44653124)
            std = (0.24703233, 0.24348505, 0.26158768)
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
        else:
            transform = transforms.ToTensor()
        self.CIFAR10 = torchvision.datasets.CIFAR10(root='./data', train=train,
                                                download=True, transform=transform)

    def __getitem__(self, index):
        data, target = self.CIFAR10[index]

        return index, data, target

    def __len__(self):
        return len(self.CIFAR10)


class BinaryCifar10(Dataset):
    def __init__(self, train=True, size=None, normalize=False, seed=0, label_noise=None):
        if normalize:
            mean = (0.49139968, 0.48215827, 0.44653124)
            std = (0.24703233, 0.24348505, 0.26158768)
            transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize(mean, std)])
        else:
            transform = transforms.ToTensor()
        self.CIFAR10 = torchvision.datasets.CIFAR10(root='./data', train=train,
                                                download=True, transform=transform)

        self.CIFAR10.targets = torch.tensor(self.CIFAR10.targets)
        class1 = 3
        class2 = 5
        subset_indices = ((self.CIFAR10.targets == class1) + (self.CIFAR10.targets == class2)).nonzero(as_tuple=False).view(-1)
        if size:
            set_seed(seed)
            subset_indices = subset_indices[torch.randint(low=0, high=len(subset_indices), size=(size, ))]
        self.CIFAR10.data = self.CIFAR10.data[subset_indices]
        self.CIFAR10.targets = self.CIFAR10.targets[subset_indices]
        self.CIFAR10.targets[self.CIFAR10.targets == class1] = 1.
        self.CIFAR10.targets[self.CIFAR10.targets == class2] = -1.

        if label_noise:
            num_noise = int(size * label_noise)
            random_labels = torch.randint(2, size=(num_noise,))
            random_labels[random_labels==0] = -1.
            self.CIFAR10.targets[:num_noise] = random_labels

    def __getitem__(self, index):
        data, target = self.CIFAR10[index]

        return index, data, target

    def __len__(self):
        return len(self.CIFAR10)


class BinaryCifar10_01(BinaryCifar10):
    def __init__(self, train=True, size=None, normalize=False, seed=0, label_noise=None):
        super(BinaryCifar10_01, self).__init__(train, size, normalize, seed, label_noise)
        self.CIFAR10.targets[self.CIFAR10.targets == -1.] = 0.