import pandas as pd
import numpy as np
import pandas as pd

# Specify the file path
file_path = 'ph_dim.csv'

# Load the CSV file into a pandas DataFrame
base_df = pd.read_csv(file_path)
base_df.groupby(['cnn_width', 'seed']).size().to_csv('ph_dim_grouped.csv')

cnn_widths = base_df['cnn_width'].unique()

print(len(cnn_widths))

seeds = [0, 1, 2]


redos = []

for cnn_width in cnn_widths:
    lr_df = base_df[base_df['cnn_width'] == cnn_width]
    
    seed_counts = dict(lr_df['seed'].value_counts())

    for seed in seeds:
        if seed in seed_counts.keys():
            if seed_counts[seed] == 0:
                redos.append({'cnn_width': cnn_width, 'seed': seed})
        else:
            redos.append({'cnn_width': cnn_width, 'seed': seed})

redo_df = pd.DataFrame(redos)
redo_df.sort_values(by=['cnn_width', 'seed'], inplace=True)
redo_df.to_csv(f'dd_redos.csv', index=False)
