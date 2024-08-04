import pandas as pd

df = pd.read_csv('ph_dim.csv')

groups = ['learning_rate', 'depth', 'seed']

df = df[df['_timestamp'] > 1715169500]
df = df[df['dataset'] == 'mnist'] 

df.to_csv('fcn_mnist.csv')

grouped = df.groupby(groups)
breakpoint()


