import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid", {"axes.facecolor": ".95"})


def cal_radmacher(K_path_list, radmacher_path=None):
    trials = len(K_path_list)
    if not radmacher_path:
        radmacher_path = K_path_list[0] + '/radmacher_' + str(trials) + '.npz'
    if os.path.exists(radmacher_path):
        radmacher = np.load(radmacher_path)
        radmacher1 = radmacher['radmacher1']
        radmacher2 = radmacher['radmacher2']
        return radmacher1, radmacher2

    B_list = []
    K_bound_lists = []
    loss_km_S_list = []
    for i, k_path in enumerate(K_path_list):
        kernel_list = np.load(k_path + '/kernel_list.npz')
        B, K_bound_list = kernel_list['B_list'], kernel_list['K_bound_list']
        B_list.append(B)
        K_bound_lists.append(K_bound_list)

        loss_km_S = np.load(k_path + '/output_km.npz')['loss_km_S']
        loss_km_S_list.append(loss_km_S)

    del kernel_list

    K_bound_lists = np.array(K_bound_lists)
    N = K_bound_lists[0].shape[-1]

    B = np.max(B_list, axis=0)
    trace_bound = np.max(np.trace(K_bound_lists, axis1=-2, axis2=-1), axis=0)

    sup = np.max(K_bound_lists, axis=0)
    inf = np.min(K_bound_lists, axis=0)
    delta = (sup - inf) / 2
    delta = delta[:, ~np.eye(delta.shape[-1], dtype=bool)]  # remove diagonal elements
    delta_sum = np.sum(delta, axis=1)
    radmacher1 = B * np.sqrt(trace_bound + delta_sum) / N

    epsilon = np.zeros(loss_km_S_list[0].shape[0])
    for i, loss_km_S in enumerate(loss_km_S_list):
        for j in range(i+1, len(loss_km_S_list)):
            dis_1norm = np.linalg.norm(loss_km_S - loss_km_S_list[j], ord=1, axis=-1)
            epsilon = np.max((epsilon, dis_1norm), axis=0)

    radmacher2 = epsilon / N

    np.savez(radmacher_path, radmacher1=radmacher1, radmacher2=radmacher2)

    return radmacher1, radmacher2



def plot_gene_gap(loss_sample_gf, loss_sample_km, loss_sample_nn, loss_train_gf, loss_km_train, loss_test_gf, acc_test_gf, radmacher, times, save_path, loss_fn):

    # plot
    loss_path = save_path + '/gene_gap.png'
    plt.figure(figsize=(15, 10), dpi=500)

    ax = plt.subplot(2, 2, 1)
    for i in range(5):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(times, loss_sample_gf[:, i], linestyle='-', color=color, label='NN GF')
        plt.plot(times, loss_sample_nn[:, i], linestyle='--', color=color, label='NN GD')
        plt.plot(times, loss_sample_km[:, i], linestyle=':', color=color, label='KM')
        if i == 0:
            plt.legend(fontsize=14)
    plt.xlabel(r'$t$', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('(a) {} loss for training samples'.format(loss_fn.capitalize()), fontsize=18)

    ax = plt.subplot(2, 2, 2)
    diff_gf_km = loss_sample_gf - loss_sample_km
    diff_gd_km = loss_sample_nn - loss_sample_km
    for i in range(5):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(times, diff_gf_km[:, i], linestyle='-', color=color, label='NN GF - KM')
        plt.plot(times, diff_gd_km[:, i], linestyle='--', color=color, label='NN GD - KM')
        if i == 0:
            plt.legend(fontsize=14)
    plt.xlabel(r'$t$', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('(b) Difference of {} loss'.format(loss_fn), fontsize=18)


    ax = plt.subplot(2, 2, 3)
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(times, loss_train_gf, linestyle='-', color=color, label='NN train loss')
    plt.plot(times, loss_km_train, linestyle='--', color=color, label='KM train loss')
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(times, loss_test_gf, linestyle='dotted', color=color, label='NN test loss')
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(times, 1 - acc_test_gf, linestyle='dotted', color=color, label='NN test error')
    plt.legend(fontsize=14)
    plt.xlabel(r'$t$', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('(c) {} loss and bound'.format(loss_fn.capitalize()), fontsize=18)

    gene_gap = loss_test_gf - loss_train_gf
    ax = plt.subplot(2, 2, 4)
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(times, gene_gap, linestyle='-', color=color, label='Generalization gap')
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(times, radmacher, linestyle='None', marker='.', color=color, label=r'$\hat{\mathcal{R}}^{gf}_{\mathcal{S}}(\mathcal{G}_{T})$')
    plt.legend(fontsize=14)
    plt.xlabel(r'$t$', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('(d) Rademacher complexity bound', fontsize=18)

    plt.tight_layout()
    plt.savefig(loss_path)
    plt.show()


def plot_bound_multi_trials(loss_sample_gf, loss_sample_km, loss_sample_nn, loss_train_gf, loss_km_train, loss_test_gf, acc_test_gf, radmacher1_trials, radmacher2_trials, N, times, save_path, loss_fn):

    # plot
    loss_path = save_path + '/bound_multi_trials.png'
    plt.figure(figsize=(15, 10), dpi=500)

    ax = plt.subplot(2, 2, 1)
    for i in range(5):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(times, loss_sample_gf[:, i], linestyle='-', color=color, label='NN GF')
        plt.plot(times, loss_sample_nn[:, i], linestyle='--', color=color, label='NN GD')
        plt.plot(times, loss_sample_km[:, i], linestyle=':', color=color, label='KM')
        if i == 0:
            plt.legend(fontsize=14)
    # ax.set_xscale("log", base=10)
    # ax.set_yscale("log", base=10)
    plt.xlabel(r'$t$', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('(a) {} loss for training samples'.format(loss_fn.capitalize()), fontsize=18)

    ax = plt.subplot(2, 2, 2)
    diff_gf_km = loss_sample_gf - loss_sample_km
    diff_gd_km = loss_sample_nn - loss_sample_km
    for i in range(5):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(times, diff_gf_km[:, i], linestyle='-', color=color, label='NN GF - KM')
        plt.plot(times, diff_gd_km[:, i], linestyle='--', color=color, label='NN GD - KM')
        if i == 0:
            plt.legend(fontsize=14)
    # ax.set_xscale("log", base=10)
    # ax.set_yscale("log", base=10)
    plt.xlabel(r'$t$', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('(b) Difference of {} loss'.format(loss_fn), fontsize=18)

    delta = 0.01
    constant = 3 * np.sqrt(np.log(2 / delta) / (2 * N))
    radmacher1, std1, trials = radmacher1_trials[-1]
    radmacher2, std2, trials = radmacher2_trials[-1]
    hinge_bound1 = loss_km_train + 2 * radmacher1 + constant
    hinge_bound2 = loss_km_train + 2 * radmacher2 + constant
    ax = plt.subplot(2, 2, 3)
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(times, loss_train_gf, linestyle='-', color=color, label='NN train loss')
    # plt.plot(times, loss_km_train, linestyle='--', color=color, label='KM train loss')
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(times, loss_test_gf, linestyle='dotted', color=color, label='NN test loss')
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(times, 1 - acc_test_gf, linestyle='dotted', color=color, label='NN test error')
    color1 = next(ax._get_lines.prop_cycler)['color']
    plt.plot(times, hinge_bound1, linestyle='None', marker='.', color=color1, label=r'Bound with $U_1, {} \ S^\prime$'.format(trials))
    ax.fill_between(times, hinge_bound1 - 2*std1, hinge_bound1 + 2*std1, color=color1, alpha=0.2)
    color2 = next(ax._get_lines.prop_cycler)['color']
    plt.plot(times, hinge_bound2, linestyle='None', marker='.', color=color2, label=r'Bound with $U_2, {} \ S^\prime$'.format(trials))
    ax.fill_between(times, hinge_bound2 - 2 * std2, hinge_bound2 + 2 * std2, color=color2, alpha=0.2)
    # ax.set_xscale("log", base=10)
    # ax.set_yscale("log", base=10)
    plt.legend(fontsize=14)
    plt.xlabel(r'$t$', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('(c) {} loss and bound'.format(loss_fn.capitalize()), fontsize=18)


    gene_gap = loss_test_gf - loss_train_gf
    ax = plt.subplot(2, 2, 4)
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(times, gene_gap, linestyle='-', color=color, label='Generalization gap')
    for radmacher1, std, trials in radmacher1_trials:
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(times, radmacher1, linestyle='-', color=color, label=r'$U_1, {} \ S^\prime$'.format(trials))
        ax.fill_between(times, radmacher1 - std, radmacher1 + std, color=color, alpha=0.2)
    for radmacher2, std, trials in radmacher2_trials:
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(times, radmacher2, linestyle='-', color=color, label=r'$U_2, {} \ S^\prime$'.format(trials))
        ax.fill_between(times, radmacher2 - std, radmacher2 + std, color=color, alpha=0.2)
    # ax.set_xscale("log", base=10)
    # ax.set_yscale("log", base=10)
    plt.legend(fontsize=14)
    plt.xlabel(r'$t$', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('(d) Rademacher complexity bound', fontsize=18)

    plt.tight_layout()
    plt.savefig(loss_path)
    plt.show()


def plot_bound_multi_trials_simple(loss_sample_gf, loss_sample_km, loss_sample_nn, loss_train_gf, loss_km_train, loss_test_gf, acc_test_gf, radmacher_trials, N, times, save_path, loss_fn):

    # plot
    loss_path = save_path + '/bound_multi_trials_simple.png'
    plt.figure(figsize=(18, 5), dpi=500)

    ax = plt.subplot(1, 3, 1)
    for i in range(5):
        color = next(ax._get_lines.prop_cycler)['color']
        plt.plot(times, loss_sample_gf[:, i], linestyle='-', color=color, label='NN GF')
        plt.plot(times, loss_sample_nn[:, i], linestyle='--', color=color, label='NN GD')
        plt.plot(times, loss_sample_km[:, i], linestyle=':', color=color, label='KM')
        if i == 0:
            plt.legend(fontsize=14)
    # ax.set_xscale("log", base=10)
    # ax.set_yscale("log", base=10)
    plt.xlabel(r'$t$', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('(a) {} loss for training samples'.format(loss_fn.capitalize()), fontsize=18)

    delta = 0.01
    constant = 3 * np.sqrt(np.log(2 / delta) / (2 * N))
    radmacher, std, trials = radmacher_trials[-1]
    hinge_bound = loss_km_train + 2 * radmacher + constant
    ax = plt.subplot(1, 3, 2)
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(times, loss_train_gf, linestyle='-', color=color, label='NN train loss')
    # plt.plot(times, loss_km, linestyle='--', color=color, label='KM train loss')
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(times, loss_test_gf, linestyle='dotted', color=color, label='NN test loss')
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(times, 1 - acc_test_gf, linestyle='dotted', color=color, label='NN test error')
    color1 = next(ax._get_lines.prop_cycler)['color']
    plt.plot(times, hinge_bound, linestyle='None', marker='.', color=color1, label=r'$L_\mu(w)$ Bound, {} $S^\prime$'.format(trials))
    # plt.plot(times, hinge_bound, linestyle='None', marker='.', color=color1, label=r'Population loss bound, {} \ S^\prime$'.format(trials))
    ax.fill_between(times, hinge_bound - 2*std, hinge_bound + 2*std, color=color1, alpha=0.2)
    plt.legend(fontsize=14)
    plt.xlabel(r'$t$', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('(b) {} loss and bound'.format(loss_fn.capitalize()), fontsize=18)


    gene_gap = loss_test_gf - loss_train_gf
    ax = plt.subplot(1, 3, 3)
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(times, gene_gap, linestyle='-', color=color, label='Generalization gap')
    for radmacher, std, trials in radmacher_trials:
        color = next(ax._get_lines.prop_cycler)['color']
        # plt.plot(times, radmacher, linestyle='-', color=color, label=r'Complexity bound, {} $S^\prime$'.format(trials))
        plt.plot(times, radmacher, linestyle='-', color=color, label=r'$\hat{\mathcal{R}}^{gf}_{\mathcal{S}}(\mathcal{G}_{T})$, ' + str(trials) + r' $\mathcal{S}^\prime$')
        ax.fill_between(times, radmacher - std, radmacher + std, color=color, alpha=0.2)
    # ax.set_xscale("log", base=10)
    # ax.set_yscale("log", base=10)
    plt.legend(fontsize=14)
    plt.xlabel(r'$t$', fontsize=18)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.title('(c) Rademacher complexity bound', fontsize=18)

    plt.tight_layout()
    plt.savefig(loss_path)
    plt.show()





if __name__ == '__main__':

    loss_fn = 'logistic'
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="mnist")
    args, unknown = parser.parse_known_args()
    if args.dataset == 'mnist':
        exp_path = 'exper/mnist_1_7_gene_gd/'
    elif args.dataset == 'mnist':
        exp_path = 'exper/cifar10_3_5_gene_gd/'
    else:
        raise ValueError('Unknown dataset.')

    ############### plot MNIST or CIFAR-10 generalization bound######################
    exp_paths = os.listdir(exp_path)
    exp_paths = sorted(exp_paths)
    trials = len(os.listdir(exp_path))
    N = 1000
    lr = 10
    K_path = ()
    for path in exp_paths:
        K_path += (exp_path + '/' + path,)
    radmacher_path = K_path[0] + '/radmacher_' + str(trials) + '.npz'
    if os.path.exists(radmacher_path):
        radmacher = np.load(radmacher_path)
        radmacher1 = radmacher['radmacher1']
        radmacher2 = radmacher['radmacher2']
    else:
        radmacher1, radmacher2 = cal_radmacher(K_path)

    radmacher = np.min((radmacher1, radmacher2), axis=0)
    save_path = K_path[0]
    output_nn = np.load(save_path + '/output_nn.npz')
    output_km = np.load(save_path + '/output_km.npz')
    output_gf = np.load(save_path + '/output_gf.npz')

    loss_sample_gf = output_gf['loss_sample_gf']
    loss_train_gf = output_gf['loss_train_gf']
    loss_test_gf = output_gf['loss_test_gf']
    acc_test_gf = output_gf['acc_test_gf']

    loss_train_nn = output_nn['loss_nn']
    loss_sample_nn = output_nn['loss_sample_nn']
    loss_test_nn = output_nn['loss_test_nn']
    acc_test_nn = output_nn['acc_test_nn']

    loss_sample_km = output_km['loss_sample_km']
    loss_km_train = output_km['loss_km_train']
    times = np.arange(len(loss_sample_gf)) * lr
    plot_gene_gap(loss_sample_gf, loss_sample_km, loss_sample_nn, loss_train_gf, loss_km_train, loss_test_gf, acc_test_gf, radmacher, times, save_path, loss_fn)


    # ############### plot mnist_1_7 with multiple numbers of S' and error bar ######################
    # need to rum 5*100 random seeds of experiments
    # N = 1000
    # trails_list = [20, 50, 100]
    # K_path0 = '/exper/mnist_1_7_gene_gd/100_1000_1000_1_101_lr10_0'
    # radmacher1_trials = []
    # radmacher2_trials = []
    # radmacher_trials = []
    # for trials in trails_list:
    #     radmacher1_repeat = []
    #     radmacher2_repeat = []
    #     radmacher_repeat = []
    #     for repeat in range(0, 5):
    #         K_path = ()
    #         for seed in range(trials*repeat, trials*(repeat+1)):
    #             K_path += (K_path0[:-1] + str(seed),)
    #
    #         radmacher_path = K_path0 + '/radmacher' + str(trials) + '_repeat' + str(repeat) +'.npz'
    #         if os.path.exists(radmacher_path):
    #             radmacher = np.load(radmacher_path)
    #             radmacher1 = radmacher['radmacher1']
    #             radmacher2 = radmacher['radmacher2']
    #         else:
    #             radmacher1, radmacher2 = cal_radmacher(K_path, radmacher_path)
    #         radmacher1_repeat.append(radmacher1)
    #         radmacher2_repeat.append(radmacher2)
    #         radmacher_repeat.append(np.min((radmacher1, radmacher2), axis=0))
    #
    #     radmacher1_trials.append((np.mean(radmacher1_repeat, axis=0), np.std(radmacher1_repeat, axis=0), trials))
    #     radmacher2_trials.append((np.mean(radmacher2_repeat, axis=0), np.std(radmacher2_repeat, axis=0), trials))
    #     radmacher_trials.append((np.mean(radmacher_repeat, axis=0), np.std(radmacher_repeat, axis=0), trials))
    #
    # save_path = K_path0
    # output_nn = np.load(save_path + '/output_nn.npz')
    # output_km = np.load(save_path + '/output_km.npz')
    # output_gf = np.load(save_path + '/output_gf.npz')
    #
    # loss_sample_gf = output_gf['loss_sample_gf']
    # loss_train_gf = output_gf['loss_train_gf']
    # loss_test_gf = output_gf['loss_test_gf']
    # acc_test_gf = output_gf['acc_test_gf']
    #
    # loss_train_nn = output_nn['loss_nn']
    # loss_sample_nn = output_nn['hinge_nn']
    # loss_test_nn = output_nn['loss_test_nn']
    # acc_test_nn = output_nn['acc_test_nn']
    #
    # loss_sample_km = output_km['hinge_km']
    # loss_km_train = output_km['loss_km_train']
    # # loss_km_val = output_km['loss_km_val']
    # # times = np.arange(1, (len(loss_nn)-1) * 10 + 2, 10)
    # times = np.arange(len(loss_sample_gf)) * 10
    # # plot_bound_multi_trials(loss_sample_gf, loss_sample_km, loss_sample_nn, loss_train_gf, loss_km_train, loss_test_gf, acc_test_gf, radmacher1_trials, radmacher2_trials, N, times, save_path, loss_fn)
    # plot_bound_multi_trials_simple(loss_sample_gf, loss_sample_km, loss_sample_nn, loss_train_gf, loss_km_train, loss_test_gf, acc_test_gf, radmacher_trials, N, times, save_path, loss_fn)









