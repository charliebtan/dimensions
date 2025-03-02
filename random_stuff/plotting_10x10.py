import pandas as pd
import matplotlib.pyplot as plt

markers_dict = {
    8: 'o',  # Circle
    16: 's',  # Square
    32: '^',  # Upward-pointing triangle
    64: 'v',  # Downward-pointing triangle
    128: '<',  # Leftward-pointing triangle
    256: '>',  # Rightward-pointing triangle
    512: '*',  # Star
    1024: 'x',  # X
    2048: 'D',  # Diamond
    4096: 'p', # Pentagon
}

def plot_acc(df, title, measure, cutoff_acc):

    df = df[df['train_acc'] > cutoff_acc]

    # df = df[df['iterations'] < 100000] # filter out points that did not reach stopping criterion in the time period

    print(len(df))

    x = abs(df['train_acc'] - df['eval_acc'])
    y = df[measure]

    for batch_size in markers_dict.keys():
        idx = df['batch_size'] == batch_size
        plt.scatter(x[idx], y[idx], marker=markers_dict[batch_size], c=df[idx]['learning_rate'])

    plt.xlabel('Accuracy gap')
    plt.ylabel(measure)

    # Add markers to the legend
    handles = []
    for batch_size, marker in markers_dict.items():
        handles.append(plt.Line2D([], [], marker=marker, linestyle='None', label=batch_size, c=(68/255, 1/255, 84/255)))

    plt.legend(handles=handles)
    plt.colorbar(label='Learning Rate')
    plt.title(title)
    plt.show()

def plot_reg(df, title, measure):

    df = df[df['iterations'] < 100000] # filter out points that did not reach stopping criterion in the time period

    print(len(df))

    x = abs(df['train_loss'] - df['test_loss'])
    y = df[measure]

    for batch_size in markers_dict.keys():
        idx = df['batch_size'] == batch_size
        plt.scatter(x[idx], y[idx], marker=markers_dict[batch_size], c=df[idx]['learning_rate'])

    plt.xlabel('Loss gap')
    plt.ylabel(measure)

    # Add markers to the legend
    handles = []
    for batch_size, marker in markers_dict.items():
        handles.append(plt.Line2D([], [], marker=marker, linestyle='None', label=batch_size, c=(68/255, 1/255, 84/255)))

    plt.legend(handles=handles)
    plt.colorbar(label='Learning Rate')
    plt.title(title)
    plt.show()

# for measure in ['ph_dim_euclidean', 'ph_dim_losses_based']:
# 
#     df = pd.read_csv('data_final/fcnn_chd_10x10.csv')
#     plot_reg(df, title='fcnn chd', measure=measure) 

# for measure in ['ph_dim_euclidean', 'ph_dim_losses_based']:
# 
#     df = pd.read_csv('data_final/alexnet_cifar10_10x10.csv')
#     df = df[df['ph_dim_losses_based'] < 20] 
#     plot_acc(df, 'alexnet cifar10', measure=measure, cutoff_acc=99) 

for measure in ['norm', 'ph_dim_losses_based']:

    df = pd.read_csv('data_final/fc_mnist_10x10.csv')
    plot_acc(df, 'fc minst', measure=measure, cutoff_acc=99)

plt.show()
