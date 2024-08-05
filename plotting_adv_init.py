import pandas as pd
import matplotlib.pyplot as plt

markers_dict = {
    'random': 'o',  # Circle
    'adv': 's',  # Square
}

colour_dict = {
    'random': (68/255, 1/255, 84/255),
    'adv': (53/255, 183/255, 120/255)
}

def plot_acc(df, title, measure, cutoff_acc, ax):

    df = df[df['train_acc'] > cutoff_acc]

    print(len(df))

    x = abs(df['train_acc'] - df['eval_acc'])
    y = df[measure]

    for init in markers_dict.keys():
        idx = df['init'] == init
        ax.scatter(x[idx], y[idx], marker=markers_dict[init], label=init) #, c=colour_dict[init])

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
