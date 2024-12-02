import os
import torch
import torch.optim.lr_scheduler

from torch.nn import CrossEntropyLoss
from torchvision.models import resnet18

from avalanche.models import SimpleMLP
from avalanche.models.resnet32 import ResNet, BasicBlock
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.checkpointing import maybe_load_checkpoint, save_checkpoint
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive, LwF, Cumulative
from avalanche.benchmarks import SplitMNIST, SplitFMNIST, SplitCIFAR10, SplitCIFAR100
from avalanche.training.self_supervised import Naive as SelfSupervisedNaive

from src.args import parse_args
from src.benchmarks import *
from src.transforms import *
from src.criterions import *
from src.loggers import *
from src.models import *
from src.plugins import *


def evaluate_strategy(strategy, eval_benchmarks):
    for benchmark, benchmark_name, current_classnames, current_bias_list in eval_benchmarks:
        print(f"Evaluating benchmark {benchmark_name}")
        strategy.eval(
            benchmark.test_stream,
            benchmark_name=benchmark_name,
            current_classnames=current_classnames,
            current_bias_list=current_bias_list,
        )


def get_benchmark(benchmark_name, seed, train_transform, eval_transform, n_experiences=1, dataset_root=None):
    base_params = {
        "n_experiences": n_experiences,
        "shuffle": True,
        "seed": seed,
        "train_transform": train_transform,
        "eval_transform": eval_transform,
    }

    if benchmark_name == "split_mnist":
        benchmark_class = SplitMNIST
        num_classes = 10
    elif benchmark_name == "split_fashion_mnist":
        benchmark_class = SplitFMNIST
        num_classes = 10
    elif benchmark_name == "split_cifar10":
        benchmark_class = SplitCIFAR10
        num_classes = 10
    elif benchmark_name == "split_cifar100":
        benchmark_class = SplitCIFAR100
        num_classes = 100
    elif benchmark_name == "concon_strict":
        benchmark_class = ConConStrict
        num_classes = 2
        base_params["dataset_root"] = dataset_root
    elif benchmark_name == "concon_disjoint":
        benchmark_class = ConConDisjoint
        num_classes = 2
        base_params["dataset_root"] = dataset_root
    elif benchmark_name == "concon_unconfounded":
        benchmark_class = ConConUnconfounded
        num_classes = 2
        base_params["dataset_root"] = dataset_root
    else:
        raise NotImplementedError

    benchmark = benchmark_class(**base_params)
    benchmark.name = benchmark_name
    return benchmark, num_classes


def get_strategy(args, model, optimizer, device, plugins, eval_plugin):
    strategy_class = None
    base_params = {
        "model": model,
        "optimizer": optimizer,
        "train_epochs": args.epochs,
        "train_mb_size": args.batch_size,
        "eval_mb_size": args.batch_size,
        "device": device,
        "plugins": plugins,
        "evaluator": eval_plugin,
    }
        
    if args.strategy == "naive":
        if args.loss_type == "self_supervised":
            strategy_class = SelfSupervisedNaive
            base_params["criterion"] = BarlowTwinsLoss()
            base_params["eval_criterion"] = torch.nn.CrossEntropyLoss()
            base_params["ss_augmentations"] = BTTrainingAugmentations(
                image_size=args.image_size)
        elif args.loss_type == "supervised":
            strategy_class = Naive
            base_params["criterion"] = CrossEntropyLoss()
        else:
            raise NotImplementedError
    elif args.strategy == "cumulative":
        strategy_class = Cumulative
        base_params["criterion"] = CrossEntropyLoss()
    elif args.strategy == "lwf":
        strategy_class = LwF
        base_params["criterion"] = CrossEntropyLoss()
        base_params["alpha"] = args.alpha
        base_params["temperature"] = args.temperature
    # ADD YOUR CUSTOM STRATEGIES HERE
    else:
        raise NotImplementedError
    
    return strategy_class(**base_params)


def run_experiment(args, seed):
    # --- CONFIG
    RNGManager.set_random_seeds(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available(
        ) and args.cuda >= 0 else "cpu"
    )

    run_name = f"{args.strategy}_w_{args.model}_on_{args.benchmark}_loss({args.loss_type})"
    run_name += f"_epochs({args.epochs})_exps({args.n_experiences})_lr({args.lr})_bs({args.batch_size})"

    # ADD CUSTOM PARAMETERS TO THE RUN NAME HERE

    if args.strategy == "lwf":
        run_name += f"_alpha({args.alpha})"
    
    if "linear_probing" in args.plugins:
        run_name += "_linear_probing"
        run_name += f"_probe_lr({args.probe_lr})_probe_epochs({args.probe_epochs})"
        
    # ADD CUSTOM PLUGIN PARAMETERS TO THE RUN NAME HERE

    output_dir = os.path.join(
        args.output_dir, args.benchmark, run_name, str(seed))
    os.makedirs(output_dir, exist_ok=True)
    logs_dir = os.path.join(output_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "checkpoint.pkl")

    shape = (args.image_size, args.image_size, 3)

    if os.path.exists(os.path.join(output_dir, "completed.txt")):
        print(f"Experiment with seed {seed} already completed")
        return

    # SAVE CONFIG
    with open(os.path.join(logs_dir, "config.txt"), "w") as f:
        f.write(str(args))
        
    # TRANSFORMS CREATION
    if args.transform == "none":
        train_transform, eval_transform = None, None
    elif args.transform == "mnist":
        train_transform, eval_transform = MNISTTransform(args.image_size)
    elif args.transform == "cifar":
        train_transform, eval_transform = CIFARTransform(args.image_size)
    elif args.transform == "barlow_twins":
        train_transform, eval_transform = BarlowTwinsTransform(args.image_size)
    else:
        raise NotImplementedError
        
    # TRAIN BENCHMARK CREATION
    benchmark, num_classes = get_benchmark(args.benchmark, seed, train_transform, eval_transform,
                              n_experiences=args.n_experiences, dataset_root=args.dataset_root)

    # EVAL BENCHMARK CREATION
    eval_benchmarks = [
        get_benchmark(benchmark_name, seed, train_transform, eval_transform,
                      n_experiences=args.n_experiences, dataset_root=args.dataset_root)[0]
        for benchmark_name in args.eval_benchmarks
    ]

    # MODEL CREATION
    if args.model == "simple_mlp":
        model = SimpleMLP(
            num_classes=num_classes,
            input_size=3 * args.image_size * args.image_size
        ).to(device)
    elif args.model == "resnet32s":
        model = ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes).to(device)
    elif args.model == "resnet32s_bt":
        model = Resnet32sBT().to(device)
    else:
        raise NotImplementedError

    # CREATE THE OPTIMIZER
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # LOGGERS
    interactive_logger = InteractiveLogger()
    csv_logger = CSVLogger(logs_dir)
    loggers = [interactive_logger, csv_logger]

    if args.wandb:
        all_configs = {
            "args": vars(args),
        }
        loggers.append(WandBLogger(
            project_name=args.project_name,
            run_name=run_name,
            config=all_configs,
        ))

    # METRICS
    metrics = []

    if "loss" in args.metrics:
        metrics.append(loss_metrics(
            minibatch=True, epoch=True, experience=True, stream=True))
    if "accuracy" in args.metrics:
        keep_track_during_training = args.loss_type != "self_supervised"
        metrics.append(accuracy_metrics(
            minibatch=keep_track_during_training,
            epoch=keep_track_during_training,
            experience=True, stream=True
        ))
    if "forgetting" in args.metrics:
        metrics.append(forgetting_metrics(experience=True, stream=True))

    eval_plugin = EvaluationPlugin(
        *metrics,
        loggers=loggers,
    )

    # CREATE THE PLUGINS
    plugins = []

    if "linear_probing" in args.plugins:
        plugins.append(LinearProbingPlugin(
            benchmark=get_benchmark(args.benchmark, seed, train_transform, eval_transform,
                                    n_experiences=1, dataset_root=args.dataset_root)[0],
            num_classes=num_classes,
            epochs=args.probe_epochs,
            lr=args.probe_lr,
        ))

    # CREATE THE STRATEGY INSTANCE
    cl_strategy = get_strategy(args, model, optimizer, device, plugins, eval_plugin)

    if args.resume_from_checkpoint:
        cl_strategy, initial_exp = maybe_load_checkpoint(
            cl_strategy, checkpoint_path)
    else:
        initial_exp = 0

    # TRAINING LOOP
    print("Starting experiment...")

    for experience in benchmark.train_stream[initial_exp:]:
        print("Start of experience ", experience.current_experience)
        cl_strategy.train(experience)
        print("Training completed")

        print("Evaluation of the current strategy:")
        for eval_benchmark in eval_benchmarks:
            cl_strategy.eval(eval_benchmark.test_stream)
        print("Evaluation completed")

        # print("Saving checkpoint")
        # save_checkpoint(cl_strategy, checkpoint_path)

    with open(os.path.join(output_dir, "completed.txt"), "w") as f:
        f.write(":)")

    if args.remove_checkpoints:
        os.system(f"rm -rf {os.path.join(output_dir, 'checkpoints')}")


def main(args):
    for seed in args.seeds:
        run_experiment(args, seed)


if __name__ == "__main__":
    args = parse_args()
    main(args)
