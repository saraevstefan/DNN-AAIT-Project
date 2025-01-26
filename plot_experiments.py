import csv
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import yaml

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

Experiment 19
{
    "data_augmentation_translate_data": false,
    "loss_function": "MSE",
    "model_name": "readerbench/RoBERT-base",
    "train_dataset_name": ["ro-sts", "biblical_01"],
    "dev_dataset_name": ["ro-sts", "biblical_01"],
    "batch_size": 64
}

Experiment 20
{
    "data_augmentation_translate_data": false,
    "loss_function": "MSE",
    "model_name": "dumitrescustefan/bert-base-romanian-uncased-v1",
    "training_method": "pipeline-biblical-sts"
}

Experiment 21
{
    "data_augmentation_translate_data": false,
    "loss_function": "MSE",
    "model_name": "readerbench/RoBERT-base",
    "training_method": "pipeline-biblical-sts"
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


def plot_table(
    experiments,
    hyperparam_keys,
    title_keys,
    dct_best_results,
    dct_hyperparams,
):
    """
    Creates a table where each row represents an experiment and each column
    shows:
      1) The hyperparameters of interest (hyperparam_keys).
      2) The best values for all metrics stored in dct_best_results[exp_id].

    This function ignores any logged per-epoch data and only relies on 
    dct_best_results.

    Parameters
    ----------
    experiments : list
        List of experiment IDs (keys in dct_hyperparams, dct_best_results).
    hyperparam_keys : list
        Which hyperparameters to display in columns (e.g. ['model_name', 'lr']).
    title_keys : list
        Which hyperparameters to include in the table title.
    dct_hyperparams : dict
        A dict of hyperparameters for each experiment.
        Example: dct_hyperparams[exp_id] = {'lr': 2e-5, 'batch_size': 16, ...}
    dct_best_results : dict
        A dict of dicts, where each sub-dict contains the best values 
        for metrics of interest.
        Example: dct_best_results[exp_id] = {
           'loss': 0.123, 'accuracy': 0.987, ...
        }
    """

    # 1. Collect all unique metric names from dct_best_results
    all_metrics = set()
    for exp_id in experiments:
        all_metrics.update(dct_best_results[exp_id].keys())

    # Sort them for a consistent column order
    all_metrics = sorted(list(all_metrics))

    # 2. Build the table columns:
    #    First, the hyperparameters of interest, 
    #    then each metric found in dct_best_results.
    column_headers = list(hyperparam_keys) + all_metrics

    # 3. Gather data (one row per experiment)
    table_rows = []
    for exp_id in experiments:
        row_data = []
        # (a) Collect hyperparameter values
        for hp_key in hyperparam_keys:
            row_data.append(dct_hyperparams[exp_id].get(hp_key, None))

        # (b) Collect metric values from dct_best_results
        for metric in all_metrics:
            row_data.append(dct_best_results[exp_id].get(metric, None))

        table_rows.append(row_data)

    # 4. Create a DataFrame for nicer formatting
    df = pd.DataFrame(table_rows, columns=column_headers, index=experiments)

    # 5. Optionally, create a title for the table using the 'title_keys'
    if title_keys:
        title_str = ", ".join(
            f"{key}={dct_hyperparams[experiments[0]].get(key, None)}"
            for key in title_keys
        )
        print("Table for experiments with:", title_str)

    # 6. Print the table
    def make_sortable(x):
        """
        Convert lists to tuples, strings to single-element tuples,
        and leave everything else as is (int, float, etc.).
        """
        if isinstance(x, list):
            return tuple(x)
        else:
            return x

    df_transformed = df.copy()
    df_transformed = df_transformed.applymap(make_sortable)
    
    print(df_transformed.sort_values(by=hyperparam_keys).to_string(float_format="%.4f"))
    print()

if __name__ == "__main__":
    PLOT_SECOND_STEP=False
    dct_results = {}
    dct_hyperparams = {}
    dct_best_results = {}
    
    default_hyperparameters = dict(
        gpus = 1,
        batch_size = 256,
        accumulate_grad_batches = 16,
        lr = 2e-05,
        model_max_length = 512,
        max_train_epochs = 20,
        model_name = "dumitrescustefan/bert-base-romanian-cased-v1",
        train_dataset_name = "ro-sts",
        dev_dataset_name = "ro-sts",
        test_dataset_name = "ro-sts",
        data_augmentation_translate_data = False,
        training_method = "simple-train"
    )

    for dirname in os.listdir("lightning_logs"):
        results, hyperparams, best_results = read_results(
            os.path.join("lightning_logs", dirname)
        )
        experiment_id = int(dirname.split("_")[1])
        dct_results[experiment_id] = results
        dct_hyperparams[experiment_id] = default_hyperparameters | hyperparams
        dct_best_results[experiment_id] = best_results

    for key in dct_best_results[0].keys():
        idx_max, _ = max(dct_best_results.items(), key=lambda x: x[1][key])

        print(f"Found best value for {key} in experiment {idx_max}")
        print(f"Best[{key}] = {dct_best_results[idx_max][key]}")
        print(f"Hyperparams {dct_hyperparams[idx_max]}")
        print()

    if PLOT_SECOND_STEP:
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

    # 3rd step
    
    # From this point on, we make 2 assessments: 
    # - AnglE loss is not that good, because our problems are in the data
    # - Live translate augmentation is not used because Google Colab does not let us do api calls to 3rd parties

    # We trained some models on the biblical data, using it alone and combining it with the ro-sts data
    # We plot how the dev metrics evolve using the same model, but different training datasets
    print("Plotting evolution of metrics during training for all experiments with the same model name, but different training sets")
    plot_curves([6, 12, 13], ['train_dataset_name'], ['model_name'], dct_results, dct_hyperparams)
    plot_curves([2, 15, 16], ['train_dataset_name'], ['model_name'], dct_results, dct_hyperparams)
    plot_table([2,6,12,13,15,16], ['train_dataset_name', 'model_name'], [], dct_best_results, dct_hyperparams)
    # We observe that training on the biblical set and evaluating on the ro-sts dataset produces the worst results, but combining the 
    # two datasets outperforms training only on ro-sts
    
    # We also trained the models using the combination of the 2 in the dev set, experimenting if a bigger and mixed dev set would perform
    # better in the test set
    # TODO: add the experiment for the newer model
    print("Plotting evolution of metrics during training for all experiments with both biblical and ro-sts, but different base models name")
    plot_curves([14, 17], ['model_name'], ['train_dataset_name'], dct_results, dct_hyperparams)
    plot_table([13,16,14,17], ['model_name', 'train_dataset_name', 'dev_dataset_name'], ['train_dataset_name'], dct_best_results, dct_hyperparams)
    
    # We also tried to see if different batch sizes influence the training
    print("Plotting evolution of metrics during training for experiments with different batch sizes")
    plot_curves([13, 18], ['batch_size'], [], dct_results, dct_hyperparams)
    plot_table([13, 18], ['batch_size'], [], dct_best_results, dct_hyperparams)
    
    
    # TODO: table
    # We implemented some multi-step training pipelines to maximize data usage
    # We also introduced a new base model and a new dataset
    # We present the following experiments:
    # 1. Train on biblical, then finetune on ro-sts to maximize performance
    # -----> biblical is a lot bigger, so we can take advantage to learn what STS means and then we transfer this to RO-STS
    # 2. Train on paraphrase, then biblical, then ro-sts
    # -----> same thing as before, but first we train for a contrastive objective to also use the data from the paraphrase set 
    print("Plotting table with metrics for different training pipelines")
    plot_table([20, 21], ['model_name', 'training_method'], [], dct_best_results, dct_hyperparams)