import pandas as pd
import matplotlib.pyplot as plt

markers_dict_reg = {
    32: 'o',  # Circle
    65: 's',  # Square
    99: '^',  # Upward-pointing triangle
    132: 'v',  # Downward-pointing triangle
    166: '<',  # Leftward-pointing triangle
    200: '>',  # Rightward-pointing triangle
}

markers_dict_acc = {
    32: 'o',  # Circle
    76: 's',  # Square
    121: '^',  # Upward-pointing triangle
    166: 'v',  # Downward-pointing triangle
    211: '<',  # Leftward-pointing triangle
    256: '>',  # Rightward-pointing triangle
}


def plot_acc(df, title, measure, cutoff_acc):

    df = df[df['train_acc'] > cutoff_acc]
    # df = df[df['iterations'] < 100000] # filter out points that did not reach stopping criterion in the time period

    print(len(df))

    x = abs(df['train_acc'] - df['eval_acc'])
    y = df[measure]

    for batch_size in markers_dict_acc.keys():
        idx = df['batch_size'] == batch_size
        plt.scatter(x[idx], y[idx], marker=markers_dict_acc[batch_size], c=df[idx]['learning_rate'])

    plt.xlabel('Accuracy gap')
    plt.ylabel(measure)

    # Add markers to the legend
    handles = []
    for batch_size, marker in markers_dict_acc.items():
        handles.append(plt.Line2D([], [], marker=marker, linestyle='None', label=batch_size, c=(68/255, 1/255, 84/255)))

    plt.legend(handles=handles)
    plt.colorbar(label='Learning Rate')
    plt.title(title)
    plt.show()

def plot_reg(df, title, measure):

    # df = df[df['iterations'] < 100000] # filter out points that did not reach stopping criterion in the time period

    print(len(df))

    x = abs(df['train_loss'] - df['test_loss'])
    y = df[measure]

    for batch_size in markers_dict_reg.keys():
        idx = df['batch_size'] == batch_size
        plt.scatter(x[idx], y[idx], marker=markers_dict_reg[batch_size], c=df[idx]['learning_rate'])

    plt.xlabel('Loss gap')
    plt.ylabel(measure)

    # Add markers to the legend
    handles = []
    for batch_size, marker in markers_dict_reg.items():
        handles.append(plt.Line2D([], [], marker=marker, linestyle='None', label=batch_size, c=(68/255, 1/255, 84/255)))

    plt.legend(handles=handles)
    plt.colorbar(label='Learning Rate')
    plt.title(title)
    plt.show()

for measure in ['norm', 'ph_dim_losses_based']:

    df = pd.read_csv('data_old/fcnn7_chd_6x6.csv')
    plot_reg(df, title='fcnn chd', measure=measure) 

for measure in ['norm', 'ph_dim_losses_based']:

    df = pd.read_csv('data_old/alexnet_cifar10_6x6.csv')
    plot_acc(df, 'alexnet cifar10', measure=measure, cutoff_acc=10)

for measure in ['norm', 'ph_dim_losses_based']:

    df = pd.read_csv('data_old/fc5_mnist_6x6.csv')
    plot_acc(df, 'fc minst', measure=measure, cutoff_acc=10)

plt.show()
