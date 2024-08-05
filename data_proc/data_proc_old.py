import pandas as pd
import numpy as np
import pandas as pd

# Specify the file path
file_path = 'data_old/fcn_california.csv'

learning_rates = np.exp(np.linspace(np.log(1e-3), np.log(1e-2), 6))
# batch_sizes = [32, 76, 121, 166, 211, 256]
batch_sizes = [32, 65, 99, 132, 166, 200]
model = 'fcn5_cali'

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

df = df[df['depth'] == 5]

df.sort_values(by=['batch_size', 'learning_rate', 'seed'], inplace=True)
df.to_csv(f'{model}_filtered.csv', index=False)

count_data = []
redos = []
drop_index = []

for learning_rate in learning_rates:

    # learning rate is a float so need to allow for error in comparison
    lr_df = df[(df['learning_rate'] >= learning_rate * 0.97) & (df['learning_rate'] <= learning_rate * 1.03)]

    for batch_size in batch_sizes:
        bs_df = lr_df[lr_df['batch_size'] == batch_size]

        seed_counts = dict(bs_df['seed'].value_counts())

        count_data.append({'batch_size': batch_size, 'learning_rate': learning_rate, 'count': len(bs_df), **seed_counts})

        seeds = [i for i in range(10)]

        for seed in seeds:
            if seed in seed_counts.keys():
                if seed_counts[seed] == 0:
                    redos.append({'model': model, 'batch_size': batch_size, 'learning_rate': learning_rate, 'seed': seed})
                elif seed_counts[seed] > 1:
                    seed_df = bs_df[bs_df['seed'] == seed]
                    drop_index.extend(seed_df.index[1:])
            else:
                redos.append({'model': model, 'batch_size': batch_size, 'learning_rate': learning_rate, 'seed': seed})

new_df = df.drop(drop_index)
new_df.to_csv(f'{model}_filtered.csv', index=False)
new_df.groupby(['batch_size', 'learning_rate']).size().reset_index(name='count').to_csv(f'{model}_group_counts.csv', index=False)

if len(redos) == 0:
    print('No redos required')
else:
    redo_df = pd.DataFrame(redos)
    redo_df.sort_values(by=['batch_size', 'learning_rate', 'seed'], inplace=True)
    redo_df.to_csv(f'{model}_redos.csv', index=False)

count_df = pd.DataFrame(count_data)
count_df.sort_values(by=['batch_size', 'learning_rate'], inplace=True)
count_df.to_csv(f'{model}_group_counts.csv', index=False)
