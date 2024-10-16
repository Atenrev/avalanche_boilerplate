# Avalanche Boilerplate
This is a boilerplate for creating a new continual learning project using the [Avalanche](https://github.com/ContinualAI/avalanche) library. It includes a basic structure for the project, as well as some useful tools and libraries.

## Getting Started
To use the code, first clone the repository:

```shell
git clone https://github.com/Atenrev/avalanche_boilerplate.git
cd avalanche_boilerplate
```

Then, assuming you have Python 3.9 set up, install the required libraries:

```shell
pip install -r requirements.txt
```

## Running the Code
To run the continual learning experiments, use the following command:

```shell
main.py [-h]
```

For example, to run a continual learning experiment on Split MNIST with a simple MLP model using lfw, use the following command:

```shell
python main.py --strategy lfw --dataset split_mnist --model simple_mlp --output_dir results/ --alpha=1.0 --lr 0.001 --batch_size 128 --epochs 4 --seeds 42 69 1714
```

This will run the experiment with seeds 42, 69 and 1714, and save the results to ``results/dataset_name/{strategy}_on_{model}_{custom_parameters}_lr({args.lr})_bs({args.batch_size})_epochs({args.epochs})/``. In this case, the results will be saved to ``results/split_mnist/lfw_on_simple_mlp_alpha(1.0)"_lr(0.001)_bs(128)_epochs(4)/``. Inside this folder, you will find a folder for each seed, and inside each of these folders you will find the logs in CSV format inside the ``logs`` folder.

## Visualizing the results

You can generate plots of the results using the ``generate_report.py`` script. For example, to generate a report for the results of the previous example, use the following command:

```shell
python generate_report.py --experiments_path results/split_mnist/lfw_on_simple_mlp_alpha(1.0)_lr(0.001)_bs(128)_epochs(4) --metrics Accuracy_On_Trained_Experiences --plot_individual_metrics --create_table --create_comparison_plots
```

This command will generate plots and tables for the specified metrics and save them in the corresponding experiment folder. The ``--experiments_path`` argument specifies the path to the experiment results, and the ``--metrics`` argument specifies the metrics to plot and compare. The ``--plot_individual_metrics`` flag will plot the metrics for each experiment individually, the --create_table flag will create a table with the final results, and the ``--create_comparison_plots`` flag will create comparison plots.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References
This repository is based on the Avalanche library. If you use this code in your research, please consider citing the following paper:
```
Carta, Antonio, et al. "Avalanche: A pytorch library for deep continual learning." Journal of Machine Learning Research 24.363 (2023): 1-6.
```