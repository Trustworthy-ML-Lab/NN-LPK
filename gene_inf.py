import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from functions.dataset import BinaryMnist, FullMnist, FullCifar10, BinaryCifar10
from functions.sub_functions import check_dir

import neural_tangents as nt
from neural_tangents import stax
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


def cal_ntk(depth, X_train, X_test, save_name):
    # FCNN
    fcnn = []
    for _ in range(depth - 1):
      fcnn += [stax.Dense(512), stax.Relu(), ]
    fcnn += [stax.Dense(1)]
    _, _, kernel_fn = stax.serial(*fcnn)

    X_train_test = torch.cat((X_train, X_test), dim=0)
    X_train_test = torch.flatten(X_train_test, 1).numpy()
    train_size = len(X_train)

    # calculate NTK
    batch_size = len(X_train_test)
    # if binary:
    #     batch_size = len(X_train_test)
    # else:
    #     batch_size = 10000
    # batch_size = N if N < 10000 else 10000
    kernel_fn_batched = nt.batch(kernel_fn, device_count=-1, batch_size=batch_size, store_on_device=False)

    ntk_all = kernel_fn_batched(X_train_test, X_train_test, 'ntk')
    ntk_all = np.array(ntk_all)

    ntk_train = ntk_all[:train_size, :][:, :train_size]
    ntk_test = ntk_all[train_size:, :][:, train_size:]
    ntk_train_test = ntk_all[:train_size, :][:, train_size:]

    check_dir('./ntk/')
    np.savez(save_name, ntk_train=ntk_train, ntk_test=ntk_test, ntk_train_test=ntk_train_test)
    print('NTK calculation done!')

    ntk_train = torch.from_numpy(ntk_train)
    ntk_test = torch.from_numpy(ntk_test)
    ntk_train_test = torch.from_numpy(ntk_train_test)

    del X_train_test, kernel_fn_batched, kernel_fn

    return ntk_train, ntk_test, ntk_train_test


def gene_inf(dataset, S_size, binary, depth):

    # train data
    if dataset == 'mnist':
        if binary:
            train_data = BinaryMnist(train=True, size=S_size, seed=0)
            test_data = BinaryMnist(train=False)
        else:
            train_data = FullMnist(train=True)
            test_data = FullMnist(train=False)
    elif dataset == 'cifar10':
        if binary:
            train_data = BinaryCifar10(train=True)
            test_data = BinaryCifar10(train=False)
        else:
            train_data = FullCifar10(train=True)
            test_data = FullCifar10(train=False)
    else:
        raise RuntimeError('Wrong dataset!')

    train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
    test_loader = DataLoader(test_data, batch_size=len(test_data), shuffle=False)
    _, X_train, Y_train = next(iter(train_loader))
    _, X_test, Y_test = next(iter(test_loader))
    if not binary:  # use half of the data
        X_train = X_train[:int(len(X_train)/2)]
        Y_train = Y_train[:int(len(Y_train) / 2)]

    Y_train = Y_train.to(device=device)
    Y_test = Y_test.to(device=device)
    if binary:
        Y_train = Y_train.float()
        Y_test = Y_test.float()
    else:
        Y_train_copy = Y_train
        Y_test_copy = Y_test
        Y_train = torch.nn.functional.one_hot(Y_train, 10).float()
        Y_test = torch.nn.functional.one_hot(Y_test, 10).float()
    N = X_train.size(0)
    M = X_test.size(0)

    ntk_path = './ntk/' + dataset + '_' + ('binary' if binary else 'all') + (str(S_size) if S_size else '') + '.npz'
    if os.path.exists(ntk_path):
        ntk_train = np.load(ntk_path)['ntk_train']
        ntk_test = np.load(ntk_path)['ntk_test']
        ntk_train_test = np.load(ntk_path)['ntk_train_test']
        ntk_train = torch.from_numpy(ntk_train)
        ntk_test = torch.from_numpy(ntk_test)
        ntk_train_test = torch.from_numpy(ntk_train_test)
    else:
        ntk_train, ntk_test, ntk_train_test = cal_ntk(depth, X_train, X_test, ntk_path)

    ntk_train = ntk_train.to(device)

    lam0, v = torch.lobpcg(ntk_train, k=1, largest=False)
    print('lam0: ', lam0)
    if lam0 < 1e-6:  # in case ntk is singular
        u, s, v = torch.svd(ntk_train)
        s_new = s
        s_new[s_new < 1e-8] += 1e-8
        ntk_train = torch.mm(torch.mm(u, torch.diag(s_new)), v.t())

    ntk_inverse = torch.inverse(ntk_train)

    if binary:
        ntk_bound = depth * torch.sqrt(Y_train @ ntk_inverse @ Y_train / N)
        ntk_bound = ntk_bound.item()
        print('ntk_bound in (Cao): ', "%.8f" % ntk_bound)


    R_infty = ntk_train.mean().sqrt() * ntk_train.abs().sum().sqrt() / N
    # R_infty = ntk_train.mean().sqrt() * ntk_train.trace().sqrt() / N
    R_infty = R_infty.item()
    print('R_infty: ', "%.8f" %  R_infty)




if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # change GPU here
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = 'mnist'
    # dataset = 'cifar10'
    S_size = 1000
    binary = True
    depth = 2

    print(dataset, 'binary:', binary)
    gene_inf(dataset, S_size, binary, depth)

    # for dataset in ['mnist', 'cifar10']:
    #     for binary in [True, False]:
    #         print(dataset, 'binary: ', binary)
    #         gene_inf(dataset, S_size, binary, depth)
