import pandas as pd
import numpy as np
import pandas as pd
import ast

model = 'alexnet_cifar10'

# Specify the file path
file_path = f'data_final/{model}_adv_init_temp.csv'

# Load the CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

for column in ['train_loss', 'test_loss', 'loss_gap', 'train_acc', 'eval_acc', 'acc_gap', 'ph_dim_euclidean', 'ph_dim_losses_based', 'std_dist', 'norm', 'step_size']:

    df.loc[df[column].str.contains('\['), column] = df.loc[df[column].str.contains('\['), column].apply(lambda x: ast.literal_eval(x)[0])

df.to_csv(f'data_final/{model}_adv_init.csv', index=False)