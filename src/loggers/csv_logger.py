import os
from typing import List, TYPE_CHECKING, Tuple, Type

import torch

from avalanche.core import SupervisedPlugin
from avalanche.evaluation.metric_results import MetricValue, TensorImage
from avalanche.logging import BaseLogger

if TYPE_CHECKING:
    from avalanche.training.templates import SupervisedTemplate

UNSUPPORTED_TYPES: Tuple[Type, ...] = (
    TensorImage,
    bytes,
)


class CSVLogger(BaseLogger, SupervisedPlugin):
    """CSV logger.

    The `CSVLogger` logs metrics into a csv file.
    Metrics are logged separately for training and evaluation in files
    training_results.csv and eval_results.csv, respectively.

    .. note::

        This Logger assumes that the user is evaluating
        on only **one** experience
        during training (see below for an example of a `train` call).

    In order to monitor the performance on held-out experience
    associated to the current training experience, set
    `eval_every=1` (or larger value) in the strategy constructor
    and pass the eval experience to the `train` method::

        `for i, exp in enumerate(benchmark.train_stream):`
            `strategy.train(exp, eval_streams=[benchmark.test_stream[i]])`

    The `strategy.eval` method should be called on the entire test stream for
    consistency, even if this is not strictly required.

    The training file header is composed of:
    metric_name, training_exp, epoch, x_plot, value

    The evaluation file header is composed of:
    metric_name, eval_exp, training_exp, value
    """

    def __init__(self, log_folder=None):
        """Creates an instance of `CSVLogger` class.

        :param log_folder: folder in which to create log files.
            If None, `csvlogs` folder in the default current directory
            will be used.
        """

        super().__init__()
        self.log_folder = log_folder if log_folder is not None else "csvlogs"
        os.makedirs(self.log_folder, exist_ok=True)
        training_file_path = os.path.join(self.log_folder, "training_results.csv")
        eval_file_path = os.path.join(self.log_folder, "eval_results.csv")

        file_exists = os.path.isfile(training_file_path)

        self.training_file = open(training_file_path, "a")
        self.eval_file = open(eval_file_path, "a")

        self.metric_vals = {}

        # current training experience id
        self.training_exp_id = None

        # if we are currently training or evaluating
        # evaluation within training will not change this flag
        self.in_train_phase = None

        if not file_exists:
            self._print_csv_headers()

    def _print_csv_headers(self):
        print(
            "benchmark_name",
            "metric_name",
            "training_exp",
            "epoch",
            "x_plot",
            "value",
            sep=",",
            file=self.training_file,
            flush=True,
        )
        print(
            "benchmark_name",
            "metric_name",
            "eval_exp",
            "training_exp",
            "value",
            sep=",",
            file=self.eval_file,
            flush=True,
        )

    def _val_to_str(self, m_val):
        if isinstance(m_val, torch.Tensor):
            return str(m_val.detach().numpy()).replace('\n', '')
        elif isinstance(m_val, float):
            return f"{m_val:.4f}"
        else:
            return str(m_val)

    def print_train_metrics(self, benchmark_name, training_exp, epoch):
        for metric_logs in self.metric_vals.values():
            for name, x, val in metric_logs:
                if isinstance(val, UNSUPPORTED_TYPES):
                    continue
                val = self._val_to_str(val)
                print(
                    benchmark_name,
                    name,
                    training_exp,
                    epoch,
                    x,
                    val,
                    sep=",",
                    file=self.training_file,
                    flush=True,
                )
        self.metric_vals = {}
        
    def log_single_metric(self, name, value, x_plot) -> None:
        if self.metric_vals.get(name) is None:
            self.metric_vals[name] = [(name, x_plot, value)]
        else:
            self.metric_vals[name].append((name, x_plot, value))

    def print_eval_metrics(self, benchmark_name, eval_exp, training_exp):
        for metric_logs in self.metric_vals.values():
            for name, x, val in metric_logs:
                if isinstance(val, UNSUPPORTED_TYPES):
                    continue

                val = self._val_to_str(val)
                print(
                    benchmark_name,
                    name,
                    eval_exp,
                    training_exp,
                    val,
                    sep=",",
                    file=self.eval_file,
                    flush=True,
                )

    def after_training_epoch(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs,
    ):
        super().after_training_epoch(strategy, metric_values, **kwargs)
        self.print_train_metrics(
            strategy.experience.benchmark.name,
            self.training_exp_id,
            strategy.clock.train_exp_epochs,
        )
        self.metric_vals = {}

    def after_eval_exp(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs,
    ):
        super().after_eval_exp(strategy, metric_values, **kwargs)

        if not self.in_train_phase:
            self.print_eval_metrics(
                strategy.experience.benchmark.name,
                strategy.experience.current_experience,
                self.training_exp_id,
            )

        self.metric_vals = {}

    def before_training_exp(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs,
    ):
        super().before_training(strategy, metric_values, **kwargs)
        self.training_exp_id = strategy.experience.current_experience

    def before_eval(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs,
    ):
        """
        Manage the case in which `eval` is first called before `train`
        """
        if self.in_train_phase is None:
            self.in_train_phase = False

    def after_eval(
            self, 
            strategy: "SupervisedTemplate",
             metric_values: List["MetricValue"], 
             **kwargs
    ):
        super().after_eval(strategy, metric_values, **kwargs)
        self.print_eval_metrics(
            strategy.experience.benchmark.name,
            strategy.experience.current_experience,
            self.training_exp_id,
        )
        self.metric_vals = {}

    def before_training(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs,
    ):
        self.in_train_phase = True

    def after_training(
        self,
        strategy: "SupervisedTemplate",
        metric_values: List["MetricValue"],
        **kwargs,
    ):
        self.in_train_phase = False

    def close(self):
        self.training_file.close()
        self.eval_file.close()


def _remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text  # or whatever


__all__ = ["CSVLogger"]