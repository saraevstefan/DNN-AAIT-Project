import os
import yaml
import json
import csv
import matplotlib.pyplot as plt

"""
Experiment 0
{
    "data_augmentation_translate_data": false,
    "loss_function": "MSE",
    "model_name": "dumitrescustefan/bert-base-romanian-cased-v1",
}

Experiment 1
{
    "data_augmentation_translate_data": false,
    "loss_function": "AnglE",
    "model_name": "dumitrescustefan/bert-base-romanian-cased-v1",
}

Experiment 2
{
    "data_augmentation_translate_data": false,
    "loss_function": "MSE",
    "model_name": "dumitrescustefan/bert-base-romanian-uncased-v1",
}

Experiment 3
{
    "data_augmentation_translate_data": false,
    "loss_function": "AnglE",
    "model_name": "dumitrescustefan/bert-base-romanian-uncased-v1",
}

Experiment 4
{
    "data_augmentation_translate_data": false,
    "loss_function": "MSE",
    "model_name": "readerbench/RoBERT-small",
}

Experiment 5
{
    "data_augmentation_translate_data": false,
    "loss_function": "AnglE",
    "model_name": "readerbench/RoBERT-small",
}

Experiment 6
{
    "data_augmentation_translate_data": false,
    "loss_function": "MSE",
    "model_name": "readerbench/RoBERT-base",
}

Experiment 7
{
    "data_augmentation_translate_data": false,
    "loss_function": "AnglE",
    "model_name": "readerbench/RoBERT-base",
}

Experiment 8
{
    "data_augmentation_translate_data": true,
    "loss_function": "MSE",
    "model_name": "dumitrescustefan/bert-base-romanian-uncased-v1"
}

Experiment 9
{
    "data_augmentation_translate_data": true,
    "loss_function": "AnglE",
    "model_name": "dumitrescustefan/bert-base-romanian-uncased-v1"
}

Experiment 10
{
    "data_augmentation_translate_data": true,
    "loss_function": "MSE",
    "model_name": "readerbench/RoBERT-base"
}

Experiment 11
{
    "data_augmentation_translate_data": true,
    "loss_function": "AnglE",
    "model_name": "readerbench/RoBERT-base"
}

Experiment 12
{
    "data_augmentation_translate_data": false,
    "loss_function": "MSE",
    "model_name": "readerbench/RoBERT-base",
    "train_dataset_name": "biblical_01"
}

Experiment 13
{
    "data_augmentation_translate_data": false,
    "loss_function": "MSE",
    "model_name": "readerbench/RoBERT-base",
    "train_dataset_name": ["ro-sts", "biblical_01"],
    "dev_dataset_name": "ro-sts"
}

Experiment 14
{
    "data_augmentation_translate_data": false,
    "loss_function": "MSE",
    "model_name": "readerbench/RoBERT-base",
    "train_dataset_name": ["ro-sts", "biblical_01"],
    "dev_dataset_name": ["ro-sts", "biblical_01"]
}

Experiment 15
{
    "data_augmentation_translate_data": false,
    "loss_function": "MSE",
    "model_name": "dumitrescustefan/bert-base-romanian-uncased-v1",
    "train_dataset_name": "biblical_01"
}

Experiment 16
{
    "data_augmentation_translate_data": false,
    "loss_function": "MSE",
    "model_name": "dumitrescustefan/bert-base-romanian-uncased-v1",
    "train_dataset_name": ["ro-sts", "biblical_01"],
    "dev_dataset_name": "ro-sts"
}

Experiment 17
{
    "data_augmentation_translate_data": false,
    "loss_function": "MSE",
    "model_name": "dumitrescustefan/bert-base-romanian-uncased-v1",
    "train_dataset_name": ["ro-sts", "biblical_01"],
    "dev_dataset_name": ["ro-sts", "biblical_01"]
}

Experiment 18
{
    "data_augmentation_translate_data": false,
    "loss_function": "MSE",
    "model_name": "readerbench/RoBERT-base",
    "train_dataset_name": ["ro-sts", "biblical_01"],
    "dev_dataset_name": "ro-sts",
    "batch_size": 64
}
"""


def read_results(experiment_folder):
    with open(os.path.join(experiment_folder, "results.json"), "r") as fd:
        best_results = json.load(fd)

    with open(os.path.join(experiment_folder, "hparams.yaml"), "r") as fd:
        hyperparams = yaml.safe_load(fd)

    try:
        results = []
        with open(os.path.join(experiment_folder, "metrics.csv"), "r") as fd:
            reader = csv.DictReader(fd)
            for row in reader:
                results.append(dict(row))
            results = results[:-2]

        dct_results = {}
        for line in results:
            _ = line.pop('epoch')
            _ = line.pop('step')
            for key, value in line.items():
                if key not in dct_results:
                    dct_results[key] = []
                if value != '':
                    value = float(value)
                dct_results[key].append(value)
        to_pop_keys = []
        for key, value in dct_results.items():
            if all(x == '' for x in value):
                to_pop_keys.append(key)
        for key in to_pop_keys:
            dct_results.pop(key)
    except Exception as e:
        dct_results = {}
        print(e)
    return dct_results, hyperparams, best_results

def plot_curves(experiments, hyperparam_keys, title_keys, dct_results, dct_hyperparams, aggregate_prefix=['dev/', 'test/']):
    # get number of graphs
    example_result = dct_results[experiments[0]]
    lst_unique_graph_names = []
    for key in example_result.keys():
        for prefix in aggregate_prefix:
            if prefix in key:
                suffix = key.split(prefix)[1]
                if suffix not in lst_unique_graph_names:
                    lst_unique_graph_names.append(suffix)

    nr_graphs = len(lst_unique_graph_names)
    # Create a figure and three subplots arranged horizontally
    _, axs = plt.subplots(nr_graphs, 1, figsize=(9, 3*nr_graphs))

    for i in range(nr_graphs):
        suffix = lst_unique_graph_names[i]
        for experiment in experiments:
            for key, value in dct_results[experiment].items():
                prefix = None
                if suffix not in key:
                    continue
                for p in aggregate_prefix:
                    if p in key:
                        prefix = p
                        break
                label = '/'.join(f"{dct_hyperparams[experiment][k]}" for k in hyperparam_keys)
                axs[i].plot(range(len(value)), value, label=label)

        title = ', '.join(f"{k}" for k in hyperparam_keys)
        if title == '':
            axs[i].set_title(f'Evolution of {p}{suffix} during training')
        else:
            axs[i].set_title(f'Evolution of {p}{suffix} during training comparing `{title}`')
        axs[i].set_xlabel('Epoch')
        axs[i].set_ylabel(f'{suffix}')
        axs[i].legend()

    plt.tight_layout()
    title = ', '.join(f"{k}={dct_hyperparams[experiment][k]}" for k in title_keys)
    print("Plot experiments with the following setup: ", title)
    plt.show()


if __name__ == "__main__":
    dct_results = {}
    dct_hyperparams = {}
    dct_best_results = {}

    for dirname in os.listdir("lightning_logs"):
        results, hyperparams, best_results = read_results(
            os.path.join("lightning_logs", dirname)
        )
        experiment_id = int(dirname.split("_")[1])
        dct_results[experiment_id] = results
        dct_hyperparams[experiment_id] = hyperparams
        dct_best_results[experiment_id] = best_results

    for key in dct_best_results[0].keys():
        idx_max, _ = max(dct_best_results.items(), key=lambda x: x[1][key])

        print(f"Found best value for {key} in experiment {idx_max}")
        print(f"Best[{key}] = {dct_best_results[idx_max][key]}")
        print(f"Hyperparams {dct_hyperparams[idx_max]}")
        print()

    print("Plotting evolution of metrics during training for all experiments with loss=MSE and no data augmentation")
    plot_curves([0,2,4,6], ['model_name'], ['loss_function'], dct_results, dct_hyperparams)
    # We observe that the most recent pretrained model (RoBERT-base) achieves the best scores
    # with loss_function=MSE of desired cosine similarity and predicted cosine similarity


    print("Plotting evolution of metrics during training for all experiments with loss=AnglE and no data augmentation")
    plot_curves([1,3,5,7], ['model_name'], ['loss_function'], dct_results, dct_hyperparams)
    # We observe that the most recent pretrained model (RoBERT-base) achieves the best scores
    # with loss_function=AnglE

    print("Plotting evolution of metrics during training for all experiments with the same model name and no data augmentation")
    plot_curves([0,1], ['loss_function'], ['model_name'], dct_results, dct_hyperparams)
    plot_curves([2,3], ['loss_function'], ['model_name'], dct_results, dct_hyperparams)
    plot_curves([4,5], ['loss_function'], ['model_name'], dct_results, dct_hyperparams)
    plot_curves([6,7], ['loss_function'], ['model_name'], dct_results, dct_hyperparams)
    # We observe that AnglE loss is better for the bert-uncased backbone, but worse for the RoBERT-base.
    # AnglE loss does not impact the training, suggesting that we can get significant improvements if we focus on data

    print("Plotting evolution of metrics during training for all experiments with the same model name and loss")
    plot_curves([2,8], ['data_augmentation_translate_data'], ['loss_function', 'model_name'], dct_results, dct_hyperparams)
    plot_curves([3,9], ['data_augmentation_translate_data'], ['loss_function', 'model_name'], dct_results, dct_hyperparams)
    plot_curves([6,10], ['data_augmentation_translate_data'], ['loss_function', 'model_name'], dct_results, dct_hyperparams)
    plot_curves([7,11], ['data_augmentation_translate_data'], ['loss_function', 'model_name'], dct_results, dct_hyperparams)
    # We observe that data augmentation greatly improves performance of the model on the dev set, regardless
    # of the model used or the loss function used
    
    print("Plotting evolution of metrics during training for all experiments with the same model name and with data augmentation")
    plot_curves([8,9], ['loss_function'], ['data_augmentation_translate_data', 'model_name'], dct_results, dct_hyperparams)
    plot_curves([10,11], ['loss_function'], ['data_augmentation_translate_data', 'model_name'], dct_results, dct_hyperparams)
    # We observe that with data augmentation, the AnglE loss function performs slightly better than MSE loss.

    # Other than that, we believe we can achieve better performances by focusing on improving the training data.

    # We plan in the next set of experiments to use the biblical dataset during training.
