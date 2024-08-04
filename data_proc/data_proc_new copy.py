import pandas as pd
import numpy as np
import pandas as pd

# Specify the file path
file_path = 'raw_data_topology.csv'

learning_rates = np.logspace(-4, -1, 10, base=10)
batch_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

# Load the CSV file into a pandas DataFrame
base_df = pd.read_csv(file_path)

for model in ['alexnet', 'fc', 'fcnn']: # fc - mnist, fcnn - califorina

    df = base_df[base_df['model'] == model]

    # Filter the DataFrame to keep rows where learning_rate is in learning_rates and batch_size is in batch_sizes
    # df = df[df['learning_rate'].isin(learning_rates) & df['batch_size'].isin(batch_sizes)]
    df = df[df['batch_size'].isin(batch_sizes)]

    # filter out old runs using timestamp (manually found the timestamps)
    if model == 'fcnn':
        df = df[df['_timestamp'] >= 1715938157]
    elif model == 'alexnet':
        df = df[df['_timestamp'] >= 1715953257]
    elif model == 'fc':
        df = df[df['_timestamp'] >= 1716096619]

    df.sort_values(by=['batch_size', 'learning_rate', 'seed'], inplace=True)
    df.to_csv(f'{model}_filtered.csv', index=False)

    count_data = []

    for learning_rate in learning_rates:
        # learning rate is a float so need to allow for error in comparison
        lr_df = df[(df['learning_rate'] >= learning_rate * 0.97) & (df['learning_rate'] <= learning_rate * 1.03)]
        for batch_size in batch_sizes:
            bs_df = lr_df[lr_df['batch_size'] == batch_size]

            seed_counts = dict(bs_df['seed'].value_counts())

            count_data.append({'batch_size': batch_size, 'learning_rate': learning_rate, 'count': len(bs_df), **seed_counts})

            seeds = [0.0, 1.0, 2.0, 3.0, 4.0] if model != 'alexnet' else [0.0, 1.0, 2

            for seed in [0.0, 1.0, 2.0, 3.0, 4.0]:

    count_df = pd.DataFrame(count_data)

    count_df.to_csv(f'{model}_group_counts.csv', index=False)
