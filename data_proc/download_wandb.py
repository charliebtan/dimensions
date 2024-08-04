import wandb
import pandas as pd
from datetime import datetime, timedelta

def download_wandb_runs(project_name):
    api = wandb.Api()

    # normal start

    wandb_runs = api.runs(project_name)

    # normal end

    ## filter by date start

    # Calculate the current time and the time N days hours ago
    now = datetime.now()
    one_day_ago = now - timedelta(days=1)
    
    # Format the times
    now_str = now.strftime("%Y-%m-%dT%H:%M:%S")
    one_day_ago_str = one_day_ago.strftime("%Y-%m-%dT%H:%M:%S")
    
    # Use the times in the api.runs method
    wandb_runs = api.runs(project_name, {"$and": [{"created_at": {"$lt": now_str, "$gt": one_day_ago_str}}]})

    ## filter by date end

    data = []
    
    for wandb_run in wandb_runs:
        run_data = wandb_run.history()
        data.append(run_data)
    
    df = pd.concat(data)
    return df

# Usage example
project_name = 'ctan/ph_dim'
df = download_wandb_runs(project_name)
df.to_csv(f"{project_name.split('/')[1]}.csv", index=False)

# def download_wandb_runs(project_name):
#     api = wandb.Api()
#     
#     data = []
#     for wandb_run in wandb_runs:
#         run_data = wandb_run.history()
#         data.append(run_data)
#     
#     df = pd.concat(data)
#     return df

# Usage example
project_name = 'ctan/ph_dim'
df = download_wandb_runs(project_name)
df.to_csv(f"{project_name.split('/')[1]}.csv", index=False)

# df = df[df['model'] == 'cnn']
# df = df[df['learning_rate'] == '0.001']
# 
# # Plotting
# plt.plot(df['cnn_width'], df['eval_acc'])
# plt.xlabel('cnn_width')
# plt.ylabel('eval_acc')
# plt.title('CNN Width vs Eval Accuracy')
# plt.show()
# 
# df.to_csv(f"{project_name.split('/')[1]}.csv", index=False)


