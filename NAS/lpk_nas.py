import os, sys, datetime, argparse
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
from functorch import make_functional_with_buffers, vmap, grad
from functorch.experimental import replace_all_batch_norm_modules_
from easydict import EasyDict as edict
from pathlib import Path
import matplotlib.pyplot as plt
lib_dir = (Path(__file__).parent / 'lib').resolve()
if str(lib_dir) not in sys.path: sys.path.insert(0, str(lib_dir))
from nas_201_api import NASBench201API as API
from datasets import get_datasets
from models import get_cell_based_tiny_net
from procedures import prepare_seed


def kaiming_normal_fanin_init(m):
  if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
    if hasattr(m, 'bias') and m.bias is not None:
      nn.init.zeros_(m.bias)
  elif isinstance(m, nn.BatchNorm2d):
    nn.init.ones_(m.weight.data)
    nn.init.constant_(m.bias.data, 0.0)


def kaiming_normal_fanout_init(m):
  if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    if hasattr(m, 'bias') and m.bias is not None:
      nn.init.zeros_(m.bias)
  elif isinstance(m, nn.BatchNorm2d):
    nn.init.ones_(m.weight.data)
    nn.init.constant_(m.bias.data, 0.0)


def init_model(model, method='kaiming_norm_fanin'):
  if method == 'kaiming_norm_fanin':
    model.apply(kaiming_normal_fanin_init)
  elif method == 'kaiming_norm_fanout':
    model.apply(kaiming_normal_fanout_init)
  return model


def cal_loss_tangent_kernel(net, X, Y, loss_fn):
    net = replace_all_batch_norm_modules_(net)
    fnet, params, buffers = make_functional_with_buffers(net)

    def compute_loss_stateless_model(params, buffers, x, y):
        output = fnet(params, buffers, x.unsqueeze(0))
        output = output[1].squeeze(0)
        loss = loss_fn(output, y)
        return loss

    # Compute J(x1)
    ft_compute_sample_grad = vmap(grad(compute_loss_stateless_model), (None, None, 0, 0))
    loss_grad = ft_compute_sample_grad(params, buffers, X, Y)
    loss_grad = [j.detach().flatten(1) for j in loss_grad]
    ltk = torch.stack([torch.matmul(j, j.T) for j in loss_grad])
    ltk = ltk.sum(0)

    return ltk


def main(seed, num_arch, dataset, N, train_epochs, lr, data_path, nas201_path, momentum):
  title = dataset + '_' + str(num_arch) + '_train' + str(train_epochs) + '_lr' + str(lr) + '_N' + str(N) \
          + '_' + ('momentum' if momentum else '') + str(seed)
  print(title)
  prepare_seed(seed)
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  train_data, test_data, xshape, class_num = get_datasets(dataset.replace('-valid', ''), data_path, -1, train_aug=True)
  train_data_test, test_data, xshape, class_num = get_datasets(dataset.replace('-valid', ''), data_path, -1, train_aug=False)
  train_loader_test = torch.utils.data.DataLoader(train_data_test, batch_size=N, shuffle=True, num_workers=0, pin_memory=True)
  X, Y = next(iter(train_loader_test))
  X = X.to(device=device)
  Y = Y.to(device=device)

  acc_convergence = []
  train_loss_converge = []
  train_loss_benchmark = [[] for _ in range(train_epochs+1)]
  train_loss_list = [[] for _ in range(train_epochs + 1)]
  train_loss_batch = [[] for _ in range(train_epochs + 1)]
  rademacher_bounds = [[] for _ in range(train_epochs+1)]
  gene_bounds = [[] for _ in range(train_epochs+1)]
  acc_list = [[] for _ in range(train_epochs+1)]

  num = 15625
  np.random.seed(seed)
  sample_indices = np.random.randint(low=0, high=num, size=num_arch)

  api = API(nas201_path)

  for index in tqdm(sample_indices):

    arch_str = api[index]
    print ('{:5d}/{:5d} : {:}'.format(index, len(api), arch_str))
    # show all information for a specific architecture
    # api.show(i)

    info = api.query_meta_info_by_index(index)  # This is an instance of `ArchResults`
    test_metrics = info.get_metrics(dataset, 'ori-test')
    acc_test = test_metrics['accuracy']
    acc_convergence.append(acc_test)
    train_metrics = info.get_metrics(dataset, 'train')
    train_loss_converge.append(train_metrics['loss'])


    config = api.get_net_config(index, dataset) # obtain the network configuration for the 123-th architecture on the CIFAR-10 dataset
    network = get_cell_based_tiny_net(edict(config)) # create the network from configurration
    network = network.to(device=device).train()
    init_model(network, 'kaiming_norm_fanout')

    loss_fn = nn.CrossEntropyLoss()
    dt = lr
    step = 0
    rademacher = torch.tensor(0., device=device)
    K = torch.zeros(size=(N, N), device=device)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, shuffle=True, num_workers=0,pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=500, shuffle=False, num_workers=0, pin_memory=True)
    if momentum:
      optimizer = torch.optim.SGD(network.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005, nesterov=True)
    else:
      optimizer = torch.optim.SGD(network.parameters(), lr=dt)
    for epoch in tqdm(range(train_epochs)):
      train_metrics = info.get_metrics(dataset, 'train', iepoch=epoch)
      train_loss = train_metrics['loss']
      train_loss_benchmark[epoch].append(train_loss)

      # test NN
      acc_test, loss_test = test_nn(network, test_loader, device)
      acc_list[epoch].append(acc_test)
      # train loss
      acc_train, loss_train = test_nn(network, train_loader_test, device)
      train_loss_list[epoch].append(loss_train)

      for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device=device)
        targets = targets.to(device=device)
        ltk_curr = cal_loss_tangent_kernel(network, X, Y, loss_fn)
        # calculate the kernel machine
        t = step * dt
        if t > 0:
          K = (ltk_last + ltk_curr) * dt / 2  # Trapezoidal rule
        ltk_last = ltk_curr
        rademacher += torch.sqrt(torch.mean(K) * torch.trace(K)) / N * np.sqrt(N/50000)   # scale to whole training set

        if step == 1:
          output_X = network(X)
          loss_X = loss_fn(output_X[1], Y)
          rademacher_bounds[epoch].append(rademacher.item())
          gene_bounds[epoch].append(loss_X.item() + 2 * rademacher.item())

        # train NN
        output = network(inputs)
        loss = loss_fn(output[1], targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step ==0:
          train_loss_batch[epoch].append(loss.item())
        step += 1

      output_X = network(X)
      loss_X = loss_fn(output_X[1], Y)
      rademacher_bounds[epoch+1].append(rademacher.item())
      gene_bounds[epoch+1].append(loss_X.item() + 2 * rademacher.item())
      train_loss_batch[epoch+1].append(loss.item())

    train_metrics = info.get_metrics(dataset, 'train', iepoch=train_epochs)
    train_loss = train_metrics['loss']
    train_loss_benchmark[train_epochs].append(train_loss)
    acc_test, loss_test = test_nn(network, test_loader, device)
    acc_list[train_epochs].append(acc_test)
    acc_train, loss_train = test_nn(network, train_loader_test, device)
    train_loss_list[train_epochs].append(loss_train)

  acc_convergence = np.array(acc_convergence)
  train_loss_converge = np.array(train_loss_converge)
  train_loss_list = np.array(train_loss_list)
  train_loss_batch = np.array(train_loss_batch)
  train_loss_benchmark = np.array(train_loss_benchmark)
  rademacher_bounds = np.array(rademacher_bounds)
  gene_bounds = np.array(gene_bounds)
  acc_list = np.array(acc_list)
  acc_select = acc_convergence[np.argmin(gene_bounds, axis=-1)]
  print('random search {}: {}'.format(num_arch, acc_select))

  save_path = 'lpk_nas/' + title + '/'
  if not os.path.exists(save_path):
          os.makedirs(save_path)
  np.savez(save_path + '/gene_acc.npz', acc_convergence=acc_convergence, train_loss_converge=train_loss_converge,
           train_loss_list=train_loss_list, train_loss_batch=train_loss_batch, train_loss_benchmark=train_loss_benchmark,
           rademacher_bounds=rademacher_bounds, gene_bounds=gene_bounds, acc_list=acc_list)

  return acc_select


@torch.no_grad()
def test_nn(fcnn, test_loader, device):
    # test
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    correct = torch.tensor(0., device=device)
    loss_test = torch.tensor(0., device=device)
    for i, (X_batch, Y_batch) in enumerate(test_loader):
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
        output = fcnn(X_batch)
        loss = loss_fn(output[1], Y_batch)
        loss_test += torch.sum(loss)
        correct += torch.sum((torch.max(output[1], dim=-1)[-1] == Y_batch).float())

    N_test = len(test_loader.dataset)
    acc_test = correct.item() / N_test * 100
    loss_test = loss_test.item() / N_test
    return acc_test, loss_test


if __name__ == '__main__':

  os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # change GPU here
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_path", type=str, default='../data/')
  parser.add_argument("--nas201_path", type=str, default='/home/yilan/dataset/NAS-Bench-201/NAS-Bench-201-v1_0-e61699.pth')
  parser.add_argument("--dataset", type=str, default='cifar10', choices=['cifar10', 'cifar100'])
  parser.add_argument('--N', default=600, type=int, help='number of training samples to compute the bound')
  parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
  parser.add_argument('--num_arch', default=100, type=int, help='number of NN architectures sampled from the benchmark')
  parser.add_argument('--train_epochs', default=2, type=int, help='number of NN architectures sampled from the benchmark')
  parser.add_argument('--momentum', default=True, type=bool, help='use momentum or not')
  parser.add_argument('--seed', default=1, type=int, help='random seed')
  args, unknown = parser.parse_known_args()
  acc = main(args.seed, args.num_arch, args.dataset, args.N, args.train_epochs, args.lr, args.data_path, args.nas201_path, args.momentum)