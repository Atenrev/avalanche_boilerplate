import os
import json
import pandas as pd
import numpy as np
import argparse
import regex as re
import copy

from typing import Union, List

import sys
from pathlib import Path
# This script should be run from the root of the project
sys.path.append(str(Path(__file__).parent.parent))

from src.common.visual import plot_line_std_graph


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()    
    parser.add_argument("--experiments_path", type=str, 
                        default="results_debug/split_fashion_mnist",)
    parser.add_argument("--only_last_experience", action='store_true', default=True,
                        help="Add this flag to only consider the last experience for each experiment during the contruction of the table")
    parser.add_argument("--filter_condition", type=Union[str, List[str]], 
                        help="Filter the experiments to plot. If empty string, all experiments are plotted.",
                        default=[
                            ""
                        ])
    parser.add_argument("--metrics", type=str, nargs='+', 
                        default=["Accuracy_On_Trained_Experiences",],
                        help="Metrics to plot and compare")
    parser.add_argument("--plot_individual_metrics", action='store_true', default=True,
                        help="Plot the metrics for each experiment individually")
    parser.add_argument("--create_table", action='store_true', default=True,
                        help="Create a table with the final results")
    parser.add_argument("--create_comparison_plots", action='store_true', default=True,
                        help="Create comparison plots")
    return parser.parse_args()


def format_experiment_name(experiment_name: str) -> str:
    # FORMAT THE EXPERIMENT NAME HERE
    return experiment_name


def get_short_name(experiment_name: str) -> str:
    if "naive" in experiment_name:
        experiment_name = "Naive"
    elif "cumulative" in experiment_name:
        experiment_name = "Joint"
    
    # DEFINE YOUR OWN SHORT NAMES HERE

    return experiment_name


def format_metric_name(metric_name: str) -> str:
    # Replace slashes with underscores
    metric_name = metric_name.replace('/', '_')
    # Remove "_eval_phase_test_stream_Task000" and "_Task000"
    metric_name = metric_name.replace('_eval_phase_test_stream', '')
    metric_name = metric_name.replace('_Task000', '')
    return metric_name


def create_summary(experiment_path: str, metrics_filter: List[str] = []):
    best_seed = None
    best_metric_value = float('inf')
    best_metrics = {}
    seeds_metrics = {}

    for seed_folder in os.listdir(experiment_path):
        seed_path = os.path.join(experiment_path, seed_folder)

        if not os.path.isdir(seed_path):
            continue

        seed_path = os.path.join(seed_path, 'logs')
        last_seed_metrics = {}

        for file_name in os.listdir(seed_path):
            if not file_name.endswith('.csv') or not file_name.startswith('eval'):
                continue

            file_path = os.path.join(seed_path, file_name)
            df = pd.read_csv(file_path)
            last_experience = df['training_exp'].max()

            for experience in df['training_exp'].unique():
                experience = int(experience)
                experience_data = df[df['training_exp'] == experience]

                for metric_name in experience_data['metric_name'].unique():
                    metric_row = experience_data[experience_data['metric_name'] == metric_name]
                    metric_value = float(metric_row.iloc[-1, -1])
                    metric_name = format_metric_name(metric_name)

                    if metrics_filter and metric_name not in metrics_filter:
                        continue

                    if metric_name not in last_seed_metrics:
                        last_seed_metrics[metric_name] = {}

                    last_seed_metrics[metric_name][experience] = metric_value

            # Update best seed logic based on a chosen metric, e.g., 'Accuracy_On_Trained_Experiences'
            chosen_metric = 'Accuracy_On_Trained_Experiences'
            if chosen_metric in last_seed_metrics and last_seed_metrics[chosen_metric][last_experience] < best_metric_value:
                best_seed = seed_folder
                best_metric_value = last_seed_metrics[chosen_metric][last_experience]
                best_metrics = last_seed_metrics

        seeds_metrics[seed_folder] = last_seed_metrics

    # Calculate mean and std for each metric
    metrics_mean_std = {}
    for metric_name in seeds_metrics[list(seeds_metrics.keys())[0]]:
        metrics_mean_std[metric_name] = {}

        for experience in seeds_metrics[list(seeds_metrics.keys())[0]][metric_name]:
            metric_values = [seeds_metrics[seed][metric_name][experience] for seed in seeds_metrics]
            metrics_mean_std[metric_name][experience] = {
                'mean': np.mean(metric_values),
                'std': np.std(metric_values),
                'sem': np.std(metric_values) / np.sqrt(len(metric_values))
            }

    # Save best seed and method to json
    with open(os.path.join(experiment_path, 'summary.json'), 'w') as f:
        json.dump({
            'best_seed': best_seed,
            'best_metrics': best_metrics,
            'metrics_mean_std': metrics_mean_std
        }, f, indent=4)

    return metrics_mean_std


def plot_metrics(experiment_path: str, metrics: dict):
    for metric_name, experiences in metrics.items():
        x = list(experiences.keys())
        y = [experiences[experience]['mean'] for experience in x]
        y_sem = [experiences[experience]['sem'] for experience in x]

        y = np.array(y)
        y_sem = np.array(y_sem)

        output_path = os.path.join(experiment_path, f"{metric_name}.png")
        plot_line_std_graph(x, y, y_sem, 'Task', metric_name, None, output_path, x_ticks=x, size=(7, 4))


def create_table(experiments_path: str, experiments: dict, only_last_experience: bool = False, filter_condition: Union[str, List[str]] = ""):
    """
    Create a table with the results of all experiments 
    and save it to a csv file and a latex file.

    The numbers are rounded to 3 decimal places.

    Table format:
    Experiment Name | Metric 1 | Metric 2 | ... | Metric N
    ------------------------------------------------------
    Experiment 1    | mean±sem | mean±sem | ... | mean±sem
    Experiment 2    | mean±sem | mean±sem | ... | mean±sem
    ...
    
    Args:
        experiment_path (str): path to the experiments
        metrics (dict): dictionary with the metrics for each experiment
            format: {experiment_name: {metric: {[experience]: {mean: float, std: float}, ...}, ...}, ...}
    """
    metrics = copy.deepcopy(experiments)
    
    if only_last_experience:
        # Update the metrics to only contain the last experience
        for experiment_name, experiment_metrics in experiments.items():
            for metric_name, experiences in experiment_metrics.items():
                last_experience = max(experiences.keys())
                metrics[experiment_name][metric_name] = experiences[last_experience]
    else:
        # Update the metrics to flatten the experiences
        new_metrics = {}

        for experiment_name, experiment_metrics in metrics.items():
            new_metrics[experiment_name] = {}

            for metric_name, experiences in experiment_metrics.items():
                for experience, values in experiences.items():
                    new_metric_name = f"{metric_name} Task {experience}"
                    new_metrics[experiment_name][new_metric_name] = values
        
        metrics = new_metrics

    # Get the metrics names
    metrics_names = set()
    for metric in metrics.values():
        metrics_names.update(metric.keys())
    metrics_names = sorted(metrics_names)

    # Identify the best values for each metric
    best_values = {}
    best_sems = {}
    for metric_name in metrics_names:
        best_value = float('inf')
        best_sem = float('inf')
        for experiment_name, experiment_metrics in metrics.items():
            if "cumulative" in experiment_name or "naive" in experiment_name:
                continue
            if metric_name not in experiment_metrics:
                continue
            mean = experiment_metrics[metric_name]['mean']
            mean = -mean if 'acc' in metric_name.lower() else mean
            if mean < best_value:
                best_value = mean
                best_sem = experiment_metrics[metric_name]['sem']

        best_values[metric_name] = -best_value if 'acc' in metric_name.lower() else best_value
        best_sems[metric_name] = best_sem

    # Given the best values, mark as best values the ones that 
    # overlap with the best values considering the sem
    multiple_best_values = {}
    for metric_name in metrics_names:
        multiple_best_values[metric_name] = []
        for experiment_name, experiment_metrics in metrics.items():
            if metric_name not in experiment_metrics:
                continue

            mean = experiment_metrics[metric_name]['mean']
            sem = experiment_metrics[metric_name]['sem']
            
            if (mean + sem >= best_values[metric_name] - best_sems[metric_name] and mean < best_values[metric_name]
                or mean - sem <= best_values[metric_name] + best_sems[metric_name] and mean > best_values[metric_name]
                or mean == best_values[metric_name]):
                multiple_best_values[metric_name].append(experiment_name)

    # Create the table
    table = []
    experiment_metrics = metrics.items()
    experiment_metrics = sorted(experiment_metrics)
    for experiment_name, experiment_metrics in experiment_metrics:
        formatted_experiment_name = format_experiment_name(experiment_name)
        row = [formatted_experiment_name]

        for metric_name in metrics_names:
            if metric_name not in experiment_metrics:
                row.append('-')
                continue

            mean = experiment_metrics[metric_name]['mean']
            sem = experiment_metrics[metric_name]['sem']
            formatted_mean = f"{mean:.3f} $\\pm$ {sem:.3f}"
            
            if experiment_name in multiple_best_values[metric_name]:
                formatted_mean = f"\\textbf{{{formatted_mean}}}"

            row.append(f"{formatted_mean}")

        table.append(row)
    table = np.array(table)

    # Create a dataframe with the table
    metrics_names = [ "$" + metric_name.upper() + "$" for metric_name in metrics_names]
    df = pd.DataFrame(table, columns=['Experiment Name'] + metrics_names)
    df = df.set_index('Experiment Name').rename_axis(None)

    # Save the dataframe to latex
    if isinstance(filter_condition, list):
        filter_condition = "comparison"

    final_table = df.to_latex(escape=False)
    final_table = f"""
\\begin{{table}}[]
    \centering
    \caption{{Caption}}
    {final_table}
    \label{{tab:my_label}}
\end{{table}}
"""
    with open(os.path.join(experiments_path, f'summary_{filter_condition}.tex'), 'w') as f:
        f.write(final_table)


def plot_metrics_comparison(experiments_path: str, experiments: dict, filter_condition: Union[str, List[str]] = ""):
    """
    Create a plot with the comparison of the stream metrics for each 
    experiment and save it to a png file.

    Args:
        experiments_path (str): path to the experiments
        metrics (dict): dictionary with the metrics for each experiment
            format: {experiment_name: {metric: {[experience]: {mean: float, std: float}, ...}, ...}, ...}
    """
    experiment_names = list(experiments.keys())
    experiment_names = sorted(experiment_names, key=lambda x: 0 if "cumulative" in x or "naive" in x else 1 if "lwf" in x else 3)

    metric_names = list(set(sum([list(experiments[en].keys()) for en in experiment_names], [])))
    metric_names = sorted(set(metric_names))

    # Create the plot
    for metric_name in metric_names:
        all_experiment_means = []
        all_experiment_sems = []
        experiments_wo_metric = []

        for experiment_name in experiment_names:
            experiment_metrics = experiments[experiment_name]
            experiment_metric_means = []
            experiment_metric_stds = []

            if metric_name not in experiment_metrics:
                experiments_wo_metric.append(experiment_name)
                continue

            for experience in experiment_metrics[metric_name]:
                mean = experiment_metrics[metric_name][experience]['mean']
                std = experiment_metrics[metric_name][experience]['sem']
                experiment_metric_means.append(mean)
                experiment_metric_stds.append(std)

            all_experiment_means.append(experiment_metric_means)
            all_experiment_sems.append(experiment_metric_stds)

        all_experiment_means = np.array(all_experiment_means)
        all_experiment_sems = np.array(all_experiment_sems)

        if isinstance(filter_condition, list):
            filter_condition = "comparison"

        output_path = os.path.join(experiments_path, f"{metric_name}_{filter_condition}.png")
        metric_name = metric_name.upper()
        y_labels = []

        for experiment_name in experiment_names:
            if experiment_name in experiments_wo_metric:
                continue
            experiment_name = get_short_name(experiment_name)
            y_labels.append(experiment_name)

        x = np.array(range(all_experiment_means.shape[1]))
        max_val = all_experiment_means.max()
        y_lim = (0, max_val + 0.2)
        plot_line_std_graph(x, all_experiment_means, all_experiment_sems, 'Task', metric_name, None, output_path, x_ticks=x+1, y_labels=y_labels, y_lim=y_lim, size=(7, 4), annotate_last=True)


if __name__ == '__main__': 
    # matplotlib.use("pgf")
    # matplotlib.rcParams.update({
    #     "pgf.texsystem": "pdflatex",
    #     'font.family': 'times new roman',
    #     'font.size': 8,
    #     'text.usetex': True,
    #     'pgf.rcfonts': False,
    #     'figure.autolayout': True,
    # })
    args = __parse_args()
    filter_conditions = args.filter_condition

    if filter_conditions is None:
        filter_conditions = ""

    if isinstance(filter_conditions, str):
        filter_conditions = [filter_conditions]

    all_experiments_results = {}

    # Iterate the experiments path and plot the metrics for each experiment
    # Each experiment might have several subexperiments
    # each one of them is a folder with the results of different seeds
    for root_experiment_name in os.listdir(args.experiments_path):
        filter_condition = [condition for condition in filter_conditions if condition in root_experiment_name]

        if args.filter_condition is not None and not filter_condition:
            continue
        
        experiment_path = os.path.join(args.experiments_path, root_experiment_name)

        if not os.path.isdir(experiment_path):
            continue

        print(f"Creating summary for {root_experiment_name}...")
        metrics_mean_std = create_summary(experiment_path, args.metrics)
        all_experiments_results[root_experiment_name] = metrics_mean_std
        
        if args.plot_individual_metrics:
            print(f"Plotting metrics for {root_experiment_name}...")
            plot_metrics(experiment_path, metrics_mean_std)

    if args.create_table:
        print(f"Creating table with final results...")
        create_table(args.experiments_path, all_experiments_results, args.only_last_experience, args.filter_condition)

    if args.create_comparison_plots:
        print(f"Creating comparison plots...")
        plot_metrics_comparison(args.experiments_path, all_experiments_results, args.filter_condition)