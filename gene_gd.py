import os, time
import argparse
import datetime
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from functorch import make_functional, vmap, grad
from torchdiffeq import odeint

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid", {"axes.facecolor": ".95"})

from models.model import FCNN
from functions.dataset import BinaryMnist_01, BinaryCifar10_01
from functions.hparams import HParam
from functions.sub_functions import check_dir


def cal_loss_tangent_kernel(fcnn, X, Y, loss_fn):
    # compute the loss tangent kernel using GPU

    fnet, params = make_functional(fcnn)

    def compute_loss_stateless_model(params, x, y):
        output = fnet(params, x.unsqueeze(0)).flatten()
        loss = loss_fn(output.squeeze(0), y)
        return loss

    # Compute J(x1)
    ft_compute_sample_grad = vmap(grad(compute_loss_stateless_model), (None, 0, 0))
    loss_grad = ft_compute_sample_grad(params, X, Y)
    loss_grad = [j.detach().flatten(1) for j in loss_grad]
    ltk = torch.stack([torch.matmul(j1, j1.T) for j1 in loss_grad])
    ltk = ltk.sum(0)

    return ltk



def train(config, device, save_path):
    width = config.model.width
    hidden_layer = config.model.hidden_layer
    epochs = config.train_net.epochs
    lr = config.train_net.lr
    train_size = config.data.train_size
    bound_data_size = config.data.bound_data_size
    seed = config.model.seed
    save_name = str(width) + '_' + str(train_size) + '_' + str(bound_data_size) + '_' + str(hidden_layer) + '_' + str(epochs) \
                + '_lr' + str(lr) + (('_noise' + str(config.data.label_noise)) if config.data.label_noise is not None else '')  + '_' + str(seed)
    print(save_name)

    if 'mnist' in config.data.dataset:
        # fcnn with fixed initialization
        fcnn = FCNN(input_size=28 * 28, width=width, hidden_layer=hidden_layer, seed=0).to(device)
        # test data
        test_data = BinaryMnist_01(train=False, normalize=config.data.normalize, label_noise=config.data.label_noise)
        # train data S' and S, bound is computed on S
        train_data = BinaryMnist_01(train=True, size=train_size, normalize=config.data.normalize, seed=seed, label_noise=config.data.label_noise)  # S' in the paper
        bound_data = BinaryMnist_01(train=True, size=bound_data_size, normalize=config.data.normalize, seed=0, label_noise=config.data.label_noise)  # S in the paper
    elif 'cifar10' in config.data.dataset:
        fcnn = FCNN(input_size=3* 32 * 32, width=width, hidden_layer=hidden_layer, seed=0).to(device)
        # test data
        test_data = BinaryCifar10_01(train=False, normalize=config.data.normalize, label_noise=config.data.label_noise)
        # train data S' and S, bound is computed on S
        train_data = BinaryCifar10_01(train=True, size=train_size, normalize=config.data.normalize, seed=seed, label_noise=config.data.label_noise)  # S' in the paper
        bound_data = BinaryCifar10_01(train=True, size=bound_data_size, normalize=config.data.normalize, seed=0, label_noise=config.data.label_noise)  # S in the paper
    else:
        raise RuntimeError('Dataset error.')

    test_loader = DataLoader(test_data, batch_size=100, shuffle=False)
    train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=False)
    bound_data_loader = DataLoader(bound_data, batch_size=len(bound_data), shuffle=False)
    _, X, Y = next(iter(train_loader))
    _, X_bound, Y_bound = next(iter(bound_data_loader))
    X = torch.cat((X, X_bound)).flatten(1)
    Y = torch.cat((Y, Y_bound))

    X = X.to(device=device)
    Y = Y.to(device=device).float()
    eval_interval = config.train_net.eval_interval

    ############# train NN with gradient flow by solving the gradient flow ODE #########################
    # with torch.no_grad():
    logistic_loss = torch.nn.BCEWithLogitsLoss(reduction='none')
    output = fcnn(X).flatten()
    loss_0 = logistic_loss(output, Y).detach()
    t_list = torch.linspace(0., epochs-1, steps=epochs) * lr
    t_list = t_list[::eval_interval].to(device)
    K_0 = torch.tensor(0., device=device).expand(X.size(0), X.size(0))
    w_0 = ()
    for param_nn in fcnn.parameters():
        w_0 += (param_nn.detach(),)
    depth = len(w_0)
    # w_0 = tuple(fcnn.parameters())
    optimizer = torch.optim.SGD(fcnn.parameters(), lr)


    class Lambda(nn.Module):
        # solve the gradient flow ODE by integrating w. At the same time, integrate LTK to get LPK
        def forward(self, t, y):
            for param_nn, param_y in zip(fcnn.parameters(), y[:depth]):
                param_nn.data.copy_(param_y.data)
            ltk = cal_loss_tangent_kernel(fcnn, X, Y, logistic_loss)
            output = fcnn(X[:train_size]).flatten()
            logistic_mean = - logistic_loss(output, Y[:train_size]).mean()
            optimizer.zero_grad()
            logistic_mean.backward()
            grad_w = ()
            for param_nn in fcnn.parameters():
                grad_w += (param_nn.grad,)
            return grad_w + (ltk, )

    start_time = time.time()
    if config.train_net.solve_ode:
        print("Start solving gradient flow ODE!")
        w_K = odeint(Lambda(), w_0 + (K_0,), t_list, method='dopri5', rtol=float(config.train_net.oed_rtol))   # ODE solver. Decrease the tolerances to speed up
        K = w_K[-1].detach()
        w_list = w_K[:-1]

        loss_sample_gf, loss_train_gf, loss_test_gf, acc_test_gf = [], [], [], []
        for t_index in range(w_list[0].size(0)):
            for layer_index, param_nn in enumerate(fcnn.parameters()):
                param_nn.data.copy_(w_list[layer_index][t_index].data)
            output = fcnn(X[:train_size]).flatten()
            logistic = logistic_loss(output, Y[:train_size])
            loss_sample_gf.append(logistic[:5].flatten().detach().cpu().numpy())
            loss_train_gf.append(logistic.mean().item())

            acc_test, loss_test = test_nn(fcnn, test_loader, device, logistic_loss)
            acc_test_gf.append(acc_test)
            loss_test_gf.append(loss_test)

        loss_sample_gf, loss_train_gf = np.array(loss_sample_gf), np.array(loss_train_gf)
        loss_test_gf, acc_test_gf = np.array(loss_test_gf), np.array(acc_test_gf)
        save_path += save_name + '/'
        check_dir(save_path)
        np.savez(save_path + '/output_gf.npz', loss_sample_gf=loss_sample_gf, loss_train_gf=loss_train_gf, loss_test_gf=loss_test_gf,
                 acc_test_gf=acc_test_gf)
    else:
        print("Skip!! solving gradient flow ODE!")
        K = K_0.expand(len(t_list), X.size(0), X.size(0))

    ############# calculate the equivalent general kernel machine #########################

    output_km = - torch.mean(K[:, :, :train_size], dim=-1) + loss_0     # sum over S'
    K_train = K[:, :train_size, :][:, :, :train_size]       # LPK of training data
    K_bound = K[:, train_size:, :][:, :, train_size:]       # LPK of bound data S

    B_list = torch.sqrt(K_train.mean(dim=(-1, -2))).cpu().numpy()
    K_bound_list = K_bound.cpu().numpy()
    loss_km_train = torch.mean(output_km[:, :train_size], dim=-1).cpu().numpy()
    loss_km_val = torch.mean(output_km[:, train_size:], dim=-1).cpu().numpy()
    loss_km_S = output_km[:, train_size:].cpu().numpy()
    loss_sample_km = output_km[:, :5].cpu().numpy()
    np.savez(save_path + '/kernel_list.npz', B_list=B_list, K_bound_list=K_bound_list)
    np.savez(save_path + '/output_km.npz', loss_sample_km=loss_sample_km, loss_km_train=loss_km_train, loss_km_val=loss_km_val,
             loss_km_S=loss_km_S)

    end_time = time.time()
    print("Time spent for computing gradient flow ODE: %4f s" % (end_time - start_time))


    ######################################## Train NN with GD #####################################################
    start_time = time.time()
    loss_nn = []
    loss_sample_nn = []
    loss_test_nn = []
    acc_test_nn = []
    # re-initialize NN
    if 'mnist' in config.data.dataset:
        # fcnn with fixed initialization
        fcnn = FCNN(input_size=28 * 28, width=width, hidden_layer=hidden_layer, seed=0).to(device)
    elif 'cifar10' in config.data.dataset:
        fcnn = FCNN(input_size=3* 32 * 32, width=width, hidden_layer=hidden_layer, seed=0).to(device)
    else:
        raise RuntimeError('Dataset error.')
    optimizer = torch.optim.SGD(fcnn.parameters(), lr)

    for epoch in tqdm(range(epochs)):

        if epoch % eval_interval == 0:
            # test whole test set
            acc_test, loss_test = test_nn(fcnn, test_loader, device, logistic_loss)
            acc_test_nn.append(acc_test)
            loss_test_nn.append(loss_test)

        # train NN only with train data S'
        output = fcnn(X[:train_size]).flatten()
        logistic = logistic_loss(output, Y[:train_size])
        hinge_mean_train = torch.mean(logistic)
        optimizer.zero_grad()
        hinge_mean_train.backward()
        optimizer.step()

        if epoch % eval_interval == 0:
            loss_nn.append(hinge_mean_train.item())
            loss_sample_nn.append(logistic[:5].flatten().detach().cpu().numpy())

    end_time = time.time()
    print("Time spent for train NN: %4f s" % (end_time - start_time))
    loss_sample_nn = np.array(loss_sample_nn)
    loss_nn = np.array(loss_nn)
    loss_test_nn = np.array(loss_test_nn)
    acc_test_nn = np.array(acc_test_nn)
    np.savez(save_path + '/output_nn.npz', loss_sample_nn=loss_sample_nn, loss_nn=loss_nn, loss_test_nn=loss_test_nn, acc_test_nn=acc_test_nn)

    # calculate spectrally-normalized bound
    w = list(fcnn.parameters())
    w[0] = w[0] / w[0].size(-1) ** 0.5
    w[1] = w[1] / w[1].size(-1) ** 0.5
    w1_2norm = torch.linalg.matrix_norm(w[0], ord=2)
    w2_2norm = torch.linalg.matrix_norm(w[1], ord=2)
    w1_2_1norm = torch.linalg.vector_norm(torch.linalg.vector_norm(w[0].T, ord=2, dim=-1), ord=1)
    w2_2_1norm = torch.linalg.vector_norm(torch.linalg.vector_norm(w[1].T, ord=2, dim=-1), ord=1)
    R_spectral = w1_2norm * w2_2norm * ( (w1_2_1norm / w1_2norm) ** (2/3) +  (w2_2_1norm / w2_2norm) ** (2/3) ) ** (3/2)
    R_spectral =  R_spectral * torch.log(torch.tensor(w[0].size(-1))) / torch.sqrt(torch.tensor(bound_data_size))
    print('Spectrally-normalized bound:', R_spectral)

    # plot
    times = np.arange(epochs) * lr
    times = times[::eval_interval]
    loss_path = save_path + save_name + '.png'
    plt.figure(figsize=(20, 5), dpi=500)

    ax = plt.subplot(1, 3, 1)
    for i in range(5):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(times, loss_sample_gf[:, i], linestyle='-', color=color, label='NN GF')
        plt.plot(times, loss_sample_nn[:, i], linestyle='--', color=color, label='NN GD')
        plt.plot(times, loss_sample_km[:, i], linestyle=':', color=color, label='KM')
        if i == 0:
            plt.legend()
    plt.xlabel(r'$t$', fontsize=18)
    plt.title('(a) Logistic loss for training samples', fontsize=18)

    ax = plt.subplot(1, 3, 2)
    diff_gf_km = loss_sample_gf - loss_sample_km
    diff_gd_km = loss_sample_nn - loss_sample_km
    for i in range(5):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(times, diff_gf_km[:, i], linestyle='-', color=color, label='NN GF - KM')
        plt.plot(times, diff_gd_km[:, i], linestyle='--', color=color, label='NN GD - KM')
        if i == 0:
            plt.legend()
    plt.xlabel(r'$t$', fontsize=18)
    plt.title('(b) Difference of logistic loss', fontsize=18)

    ax = plt.subplot(1, 3, 3)
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(times, loss_nn, linestyle='-', color=color, label='NN train loss')
    plt.plot(times, loss_km_train, linestyle='--', color=color, label='KM train loss')
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(times, loss_test_nn, linestyle='dotted', color=color, label='NN test loss')
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(times, 1 - acc_test_nn, linestyle='dotted', color=color, label='NN test error')
    plt.legend()
    plt.xlabel(r'$t$', fontsize=18)
    plt.title('(c) Logistic loss', fontsize=18)

    plt.tight_layout()
    plt.savefig(loss_path)
    plt.show()



@torch.no_grad()
def test_nn(fcnn, test_loader, device, loss_fn):
    # test
    correct = torch.tensor(0., device=device)
    loss_test = torch.tensor(0., device=device)
    with torch.no_grad():
        for i, (ids_batch, X_batch, Y_batch) in enumerate(test_loader):
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device).float()
            output = fcnn(X_batch).flatten()
            correct += torch.sum((torch.sigmoid(output).round() == Y_batch).float())
            loss = loss_fn(output, Y_batch)
            loss_test += torch.sum(loss)

    N_test = len(test_loader.dataset)
    acc_test = correct.item() / N_test
    loss_test = loss_test.item() / N_test
    return acc_test, loss_test




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/mnist.yaml")
    args, unknown = parser.parse_known_args()
    config = HParam(args.config)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.data.gpu)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device, config.data.dataset)
    print(config)

    if config.data.label_noise is not None:
        save_path = 'exper/' + config.data.dataset + '_gd_noise/' + '/'
    else:
        save_path = 'exper/' + config.data.dataset + '_gene_gd/'

    for seed in range(0, 20):
        config.model.seed = seed
        train(config, device, save_path)
