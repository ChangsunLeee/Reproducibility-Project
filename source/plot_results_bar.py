import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

base_folder_path = r'C:\Users\hp\OneDrive\바탕 화면\deeplearning\results\ablation'  #폴더경로

common_names = ['Tmall', 'diginetica', 'Nowplaying', 'RetailRocket']

folder_mapping = {
    'noposition': 'DHCN-P',
    'position': 'DHCN',
    'session embedding': 'DHCN-NA'
}

results = []

for sub_folder in os.listdir(base_folder_path):
    sub_folder_path = os.path.join(base_folder_path, sub_folder)
    if os.path.isdir(sub_folder_path):
        for file_name in os.listdir(sub_folder_path):
            if file_name.endswith('.csv'):
                file_path = os.path.join(sub_folder_path, file_name)

                df = pd.read_csv(file_path)
                
                if 'Recall@20' in df.columns and 'MRR20' in df.columns and 'Best Epoch for Recall@20' in df.columns and 'Best Epoch for MRR20' in df.columns:
                    best_epoch_recall = df['Best Epoch for Recall@20'].max()
                    best_epoch_mrr = df['Best Epoch for MRR20'].max()
                    
                    recall20_best_value = df.loc[df['Best Epoch for Recall@20'] == best_epoch_recall, 'Recall@20'].max()
                    mrr20_best_value = df.loc[df['Best Epoch for MRR20'] == best_epoch_mrr, 'MRR20'].max()
                    
                    for name in common_names:
                        if name in file_name:
                            method = folder_mapping.get(sub_folder, sub_folder)
                            results.append({
                                'File Name': os.path.splitext(file_name)[0],
                                'Dataset': name,
                                'Method': method,
                                'Recall@20 Best': recall20_best_value,
                                'MRR20 Best': mrr20_best_value
                            })
                            break

results_df = pd.DataFrame(results)

results_df.sort_values(by=['Dataset', 'Method'], ascending=[True, True], inplace=True)

print(results_df)

datasets = results_df['Dataset'].unique()
methods = results_df['Method'].unique()
n_datasets = len(datasets)
width = 0.2
x_labels = ['Recall@20 Best', 'MRR@20 Best']

fig, axes = plt.subplots(nrows=1, ncols=n_datasets, figsize=(20, 5), sharey=True)

for i, dataset in enumerate(datasets):
    ax = axes[i]
    dataset_df = results_df[results_df['Dataset'] == dataset]
    
    x = np.arange(len(x_labels))
    
    for j, method in enumerate(methods):
        method_df = dataset_df[dataset_df['Method'] == method]
        if not method_df.empty:
            recall_best = method_df['Recall@20 Best'].values[0]
            mrr_best = method_df['MRR20 Best'].values[0]
            
            ax.bar(x + j * width, [recall_best, mrr_best], width, label=method)
    
    ax.set_title(dataset)
    ax.set_xticks(x + width)
    ax.set_xticklabels(x_labels)
    
    if i == 0:
        ax.set_ylabel('Performance %')
    ax.legend() 

plt.tight_layout()
plt.savefig(r'C:\Users\hp\OneDrive\바탕 화면\deeplearning\results\performance_graphs.png')  #저장위치
plt.show()

print("결과가 성공적으로 저장되었습니다.")
