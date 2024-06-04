import os
import pandas as pd
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

base_folder_path = r'C:\Users\hp\OneDrive\바탕 화면\deeplearning\results\Beta'  # 상위폴더경로

common_names = ['Tmall', 'diginetica', 'Nowplaying', 'RetailRocket']  # 데이터셋 생략가능

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
                            folder_number = re.findall(r"\d+\.\d+|\d+", sub_folder)[0]
                            results.append({
                                'File Name': os.path.splitext(file_name)[0],
                                'Dataset': name,
                                'Folder': folder_number,
                                'Recall@20 Best': recall20_best_value,
                                'MRR20 Best': mrr20_best_value
                            })
                            break

results_df = pd.DataFrame(results)

results_df.sort_values(by=['File Name', 'Dataset'], ascending=[True, True], inplace=True)
print(results_df)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

for key, grp in results_df.groupby(['Dataset']):
    axes[0].plot(grp['Folder'], grp['Recall@20 Best'], label=key, marker='o')
axes[0].set_title('Recall@20 Best Epoch')
axes[0].set_xlabel(os.path.basename(base_folder_path))
axes[0].set_ylabel('Performance %')
axes[0].legend(loc='best')
axes[0].yaxis.set_major_locator(ticker.MultipleLocator(5))  # 간격

for key, grp in results_df.groupby(['Dataset']):
    axes[1].plot(grp['Folder'], grp['MRR20 Best'], label=key, marker='o')
axes[1].set_title('MRR@20 Best Epoch')
axes[1].set_xlabel(os.path.basename(base_folder_path))
axes[1].legend(loc='best')
axes[1].yaxis.set_major_locator(ticker.MultipleLocator(1))  

plt.tight_layout()
plt.savefig(r'C:\Users\hp\OneDrive\바탕 화면\deeplearning\results\performance_graphs.png')  # 그래프 저장위치
plt.show()

print("결과가 저장되었습니다.")
