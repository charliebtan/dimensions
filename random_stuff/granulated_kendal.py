import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

df = pd.read_csv('data_old/fc5_mnist_6x6.csv')

batch_sizes = df['batch_size'].unique()
assert len(batch_sizes) == 6
print(batch_sizes)

learning_rates = df['learning_rate'].unique()
assert len(learning_rates) == 6 
print(learning_rates)

bs_kendals = 0
lr_kendals = 0

for batch_size in batch_sizes:

    df_bs = df[df['batch_size'] == batch_size]

    kendal = kendalltau(df_bs['ph_dim_losses_based'], abs(df_bs['train_acc'] - df_bs['eval_acc'])).statistic

    bs_kendals += kendal / len(batch_sizes)

for learning_rate in learning_rates:

    df_lr = df[df['learning_rate'] == learning_rate]

    kendal = kendalltau(df_lr['ph_dim_losses_based'], abs(df_lr['train_acc'] - df_lr['eval_acc'])).statistic

    lr_kendals += kendal / len(learning_rates)

granulated_kendal = (lr_kendals + bs_kendals) / 2

print(granulated_kendal)


# for measure in ['norm', 'ph_dim_losses_based']:
# 
#     df = pd.read_csv('data_old/alexnet_cifar10_6x6.csv')
# 
# for measure in ['norm', 'ph_dim_losses_based']:
# 
#     df = pd.read_csv('data_old/fc7_mnist_6x6.csv')
# 
# plt.show()
