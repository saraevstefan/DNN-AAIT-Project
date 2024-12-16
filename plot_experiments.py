import os
import yaml
import json
import csv

def read_results(experiment_folder):
    with open(os.path.join(experiment_folder, 'results.json'), 'r') as fd:
        best_results = json.load(fd)
    
    with open(os.path.join(experiment_folder, 'hparams.yaml'), 'r') as fd:
        hyperparams = yaml.safe_load(fd)
    
    results = []
    with open(os.path.join(experiment_folder, 'hparams.yaml'), 'r') as fd:
            reader = csv.DictReader(fd)
            for row in reader:
                results.append(dict(row))
    
    return results, hyperparams, best_results

if __name__ == "__main__":
    lst_results = []
    lst_hyperparams = []
    lst_best_results = []
    
    for dirname in os.listdir('lightning_logs'):
        results, hyperparams, best_results = read_results(os.path.join('lightning_logs', dirname))
        lst_results.append(results)
        lst_hyperparams.append(hyperparams)
        lst_best_results.append(best_results)
    
    for key in lst_best_results[0].keys():
        idx_max, _ = max(enumerate(lst_best_results), key=lambda x: x[1][key])
        
        print(f"Found best value for {key} in experiment {idx_max}")
        print(f"Best[{key}] = {lst_best_results[idx_max][key]}")
        print(f"Hyperparams {lst_hyperparams[idx_max]}")
        print()
        