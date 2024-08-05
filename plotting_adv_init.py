import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

markers_dict = {
    'random': 'o',  # Circle
    'adv': 's',  # Square
}

def plot_acc(df, title, measure, cutoff_acc, ax):

    df = df[df['train_acc'] > cutoff_acc]

    print(len(df))

    x = abs(df['train_acc'] - df['eval_acc'])
    y = df[measure]

    for init in markers_dict.keys():
        idx = df['init'] == init
        ax.scatter(x[idx], y[idx], marker=markers_dict[init], label=init)

        # remove the code for adding the stds - not sure it was formally correc

        # mean_x = x[idx].mean()
        # mean_y = y[idx].mean()

        # std_x = 3 * x[idx].std()
        # std_y = 3 * y[idx].std()

        # theta = np.linspace(0, 2*np.pi, 100)
        # a = std_x
        # b = std_y
        # ellipse_x = mean_x + a * np.cos(theta)
        # ellipse_y = mean_y + b * np.sin(theta)

        # ax.fill(ellipse_x, ellipse_y, alpha=0.3)

    ax.set_xlabel('Accuracy Gap')
    ax.set_ylabel(measure.replace('_', ' '))

    ax.legend()
    ax.set_title(title)

fig, axs = plt.subplots(2, 3, figsize=(12, 8))

for i, measure in enumerate(['ph_dim_euclidean', 'ph_dim_losses_based']):
    df = pd.read_csv('data_final/alexnet_cifar10_adv_init.csv')
    plot_acc(df, 'alexnet cifar10', measure=measure, cutoff_acc=99, ax=axs[i, 0])

    df = pd.read_csv('data_final/cnn_cifar10_adv_init.csv')
    plot_acc(df, 'cnn cifar10', measure=measure, cutoff_acc=99, ax=axs[i, 1])

    df = pd.read_csv('data_final/fc_mnist_adv_init.csv')
    plot_acc(df, 'fc minst', measure=measure, cutoff_acc=99, ax=axs[i, 2])

plt.tight_layout()
plt.show()
