import argparse
import pickle
import time
from util import Data, split_validation
from model import *
import os

import csv
from tqdm import tqdm  # Import tqdm library

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Nowplaying', help='dataset name: diginetica/Nowplaying/Tmall/RetailRocket/sample')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--embSize', type=int, default=100, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=int, default=3, help='the number of layer used') # Changed to int for iteration
parser.add_argument('--beta', type=float, default=0.01, help='ssl task maginitude')
parser.add_argument('--filter', type=bool, default=False, help='filter incidence matrix')

opt = parser.parse_args()
print(opt)

def main():
    base_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(base_path, '..', 'datasets', opt.dataset)
    
    train_path = os.path.join(dataset_path, 'train.txt')
    test_path = os.path.join(dataset_path, 'test.txt')
    
    for layer in range(1, 6):
        results_path = os.path.join(base_path, '..', 'results', f'layer{layer}')
        os.makedirs(results_path, exist_ok=True)
        results_file_path = os.path.join(results_path, f'{opt.dataset}_results.csv')

        with open(train_path, 'rb') as f:
            train_data = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_data = pickle.load(f)

        print('current path: ', os.getcwd())

        if opt.dataset == 'diginetica':
            n_node = 43097
        elif opt.dataset == 'Tmall':
            n_node = 40727
        elif opt.dataset == 'Nowplaying':
            n_node = 60416
        elif opt.dataset == 'RetailRocket':
            n_node = 36968    
        else:
            n_node = 309

        test_data = Data(test_data, shuffle=True, n_node=n_node)
        train_data = Data(train_data, shuffle=True, n_node=n_node)

        model = trans_to_cuda(DHCN(adjacency=train_data.adjacency, n_node=n_node, lr=opt.lr, l2=opt.l2, beta=opt.beta, layers=layer, emb_size=opt.embSize, batch_size=opt.batchSize, dataset=opt.dataset))

        top_K = [5, 10, 20]
        best_results = {}
        for K in top_K:
            best_results[f'epoch{K}'] = [0, 0]
            best_results[f'metric{K}'] = [0, 0]

        with open(results_file_path, 'w', newline='') as results_file:
            csv_writer = csv.writer(results_file)
            header_row = ['Epoch']
            for K in top_K:
                header_row.extend([f'Train Loss', f'Recall@{K}', f'MRR{K}', f'Best Epoch for Recall@{K}', f'Best Epoch for MRR{K}'])
            csv_writer.writerow(header_row)

            progress_bar = tqdm(desc='Epochs', total=opt.epoch)

            for epoch in range(opt.epoch):
                row_data = [epoch]
                metrics, total_loss = train_test(model, train_data, test_data, progress_bar)
                for K in top_K:
                    metrics[f'hit{K}'] = np.mean(metrics[f'hit{K}']) * 100
                    metrics[f'mrr{K}'] = np.mean(metrics[f'mrr{K}']) * 100
                    if best_results[f'metric{K}'][0] < metrics[f'hit{K}']:
                        best_results[f'metric{K}'][0] = metrics[f'hit{K}']
                        best_results[f'epoch{K}'][0] = epoch
                    if best_results[f'metric{K}'][1] < metrics[f'mrr{K}']:
                        best_results[f'mrr{K}'][1] = metrics[f'mrr{K}']
                        best_results[f'epoch{K}'][1] = epoch
                    row_data.extend([total_loss, metrics[f'hit{K}'], metrics[f'mrr{K}'], best_results[f'epoch{K}'][0], best_results[f'epoch{K}'][1]])
                csv_writer.writerow(row_data)
            progress_bar.close()

if __name__ == '__main__':
    main()
