import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('data_final/cnn_cifar100_double_descent_0.01.csv')

df = df[['train_acc', 'eval_acc', 'ph_dim_euclidean', 'ph_dim_losses_based', 'cnn_width', 'norm']]


def plot_dd(df, ax, measure):

    df = df[df['ph_dim_losses_based'] > 0] # am filtering out that bad point! - need to discuss

    grouped = df.groupby(['cnn_width'])

    means = grouped.mean()
    stds = grouped.std()

    x = means.index

    y1 = means['eval_acc']
    y1_std = stds['eval_acc']

    ax1 = ax

    ax1.plot(x, y1, color='tab:blue')
    ax1.fill_between(x, y1 - y1_std, y1 + y1_std, color='tab:blue', alpha=0.3)  # Add shaded error bars
    
    ax1.set_xlabel('cnn width')
    ax1.set_ylabel('eval acc', color='tab:blue')

    y2 = means[measure]
    y2_std = stds[measure]  

    ax2 = ax1.twinx()

    ax2.plot(x, y2, color='tab:red')
    ax2.fill_between(x, y2 - y2_std, y2 + y2_std, color='tab:red', alpha=0.3)  # Add shaded error bars
    ax2.set_ylabel(measure.replace('_', ' '), color='tab:red')

fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

plot_dd(df, axs[0], 'ph_dim_euclidean')
plot_dd(df, axs[1], 'ph_dim_losses_based')

plt.tight_layout()
plt.show()
