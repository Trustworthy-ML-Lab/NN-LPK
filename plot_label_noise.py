import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("darkgrid", {"axes.facecolor": ".95"})
from cal_bound import cal_radmacher

if __name__ == '__main__':
    noise_list = [0, 0.2, 0.4, 0.6, 0.8, 1]
    exp_path_noise = 'exper/mnist_1_7_gd_noise/100_1000_1000_1_2001_lr10_noise'

    lost_train_list = []
    lost_test_list = []
    error_test_list =[]
    radmacher_list1 = []
    radmacher_list2 = []
    gene_bound1 = []
    gene_bound2 = []
    trials = 20
    N = 1000
    loss_fn = 'logistic'
    for label_noise in noise_list:
        K_path_1000 = ()
        for seed in range(trials):
            K_path_1000 += (exp_path_noise + str(label_noise) + '_' + str(seed),)
        K_path = K_path_1000
        radmacher1, radmacher2 = cal_radmacher(K_path)
        radmacher_list1.append(radmacher1[-1])
        radmacher_list2.append(radmacher2[-1])

        save_path = K_path[1]
        output_nn = np.load(save_path + '/output_nn.npz')
        output_km = np.load(save_path + '/output_km.npz')

        loss_test_nn = output_nn['loss_test_nn']
        loss_km_train = output_km['loss_km_train']
        lost_train_list.append(loss_km_train[-1])
        lost_test_list.append(loss_test_nn[-1])

    radmacher_bound = np.min((radmacher_list1, radmacher_list2), axis=0)
    print('radmacher_list1: ', radmacher_list1)
    print('radmacher_list2: ', radmacher_list2)


    fig, ax = plt.subplots(figsize=(8, 6), dpi=500)
    color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(noise_list, np.array(lost_test_list) - np.array(lost_train_list), linestyle='-', marker='.', color=color, label='Generalization gap')
    color = next(ax._get_lines.prop_cycler)['color']
    ax.plot(noise_list, radmacher_bound, linestyle='-', marker='.', color=color, label=r'$\hat{\mathcal{R}}^{gf}_{\mathcal{S}}(\mathcal{G}_{T})$')
    ax.set_xlabel('Portion of label noise', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.legend(fontsize=18)

    # plt.title('Generalization bound with label noise', fontsize=18)
    plt.tight_layout()
    plt.savefig('label_noise.png', bbox_inches='tight')
    plt.show()