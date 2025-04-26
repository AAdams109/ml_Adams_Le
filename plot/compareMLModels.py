#Imports 
import pandas as pd
import matplotlib.pyplot as plt

#Load the Data 
datasets = {
    "CNN": [pd.read_csv('datasets/cnn1_dataset1.csv'),  pd.read_csv('datasets/cnn1_dataset2.csv'),  pd.read_csv('datasets/cnn1_dataset3.csv')],
    "LSTM": [pd.read_csv('datasets/lstm2_dataset1.csv'), pd.read_csv('datasets/lstm2_dataset2.csv'), pd.read_csv('datasets/lstm2_dataset3.csv')],
    "BERT": [pd.read_csv('datasets/bert_dataset1.csv'), pd.read_csv('datasets/bert_dataset2.csv'), pd.read_csv('datasets/bert_dataset3.csv')]
    }

#Plot the Models 
def compare_models(dataset_index, metric, ylabel, title):
    plt.figure(figsize=(8,5))
    for model_name, model_datasets in datasets.items():
        df = model_datasets[dataset_index]
        plt.plot(df['epoch'], df[metric], label=model_name)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(f"{title} (Dataset {dataset_index + 1})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#Display Model Tables
def display_stats(df, model_name, dataset_index):
    print(f"\n {model_name} - Dataset {dataset_index + 1}")
    print(df.round(2).to_string(index=False))
    stats = df.select_dtypes(include='number').drop(columns=['epoch']).agg(['mean', 'std']).round(2)
    print("\nMean:")
    print(stats.loc['mean'].to_string())
    print("\nStandard Deviation:")
    print(stats.loc['std'].to_string())

#Plots 
metrics = [
    ('accuracy', 'Accuracy', 'Accuracy Comparison'), 
    ('f1_score', 'F1 Score', 'F1 Score Comparison'),
    ('loss', 'Loss', 'Loss Comparison'), 
    ('memory', 'Memory (MB)', 'Memory Usage Comparison'), 
    ('train_time', 'Training Time(s)', 'Training Time Comparison')
    ]

for dataset_idx in range(3):
    for metric, ylabel, title in metrics:
        compare_models(dataset_idx, metric, ylabel, title)
    for model_name, model_datasets in datasets.items():
        display_stats(model_datasets[dataset_idx], model_name, dataset_idx)
