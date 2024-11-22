import os
import torch
import torch.optim.lr_scheduler

from torch.nn import CrossEntropyLoss
from torchvision.models import resnet18

from avalanche.models import SimpleMLP
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.checkpointing import maybe_load_checkpoint, save_checkpoint
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.supervised import Naive, LwF, Cumulative
from avalanche.benchmarks import SplitMNIST, SplitFMNIST, SplitCIFAR10, SplitCIFAR100
from avalanche.training.self_supervised import Naive as SelfSupervisedNaive

from src.args import parse_args
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


def run_experiment(args, seed):
    # --- CONFIG
    RNGManager.set_random_seeds(seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device(
        f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu"
    )

    run_name = f"{args.strategy}_on_{args.model}"

    if args.strategy == "lwf":
        run_name += f"_alpha({args.alpha})"

    # ADD CUSTOM PARAMETERS TO THE RUN NAME HERE

    run_name += f"_lr({args.lr})_bs({args.batch_size})_epochs({args.epochs})" 

    output_dir = os.path.join(args.output_dir, args.benchmark, run_name, str(seed))
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

    # MODEL CREATION
    num_classes = 10 if args.benchmark != "cifar100" else 100

    if args.model == "simple_mlp":
        model = SimpleMLP(
            num_classes=num_classes, 
            input_size=3 * args.image_size * args.image_size
        ).to(device)  
    elif args.model == "resnet18":
        model = resnet18().to(device)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif args.model == "resnet18_bt":
        model = Resnet18BT().to(device)
    else:
        raise NotImplementedError

    # CREATE THE OPTIMIZER
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)   

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

    # BENCHMARK CREATION      
    if args.benchmark == "split_mnist":
        benchmark_class = SplitMNIST
    elif args.benchmark == "split_fashion_mnist":
        benchmark_class = SplitFMNIST
    elif args.benchmark == "split_cifar10":
        benchmark_class = SplitCIFAR10
    elif args.benchmark == "split_cifar100":
        benchmark_class = SplitCIFAR100
    else:
        raise NotImplementedError

    benchmark = benchmark_class(
        n_experiences=args.n_experiences,
        shuffle=True,
        seed=seed,
        train_transform=train_transform,
        eval_transform=eval_transform,
    )

    # METRICS AND LOGGERS
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

    metrics = []

    if "loss" in args.metrics:
        metrics.append(loss_metrics(minibatch=True, epoch=True, experience=True, stream=True))
    if "accuracy" in args.metrics:
        metrics.append(accuracy_metrics(minibatch=True, epoch=True, experience=True, stream=True))
    if "forgetting" in args.metrics:
        metrics.append(forgetting_metrics(experience=True, stream=True))

    eval_plugin = EvaluationPlugin(
        *metrics,
        # ADD YOUR CUSTOM METRICS HERE
        loggers=loggers,
    )

    # CREATE THE PLUGINS
    plugins = []

    if "linear_probing" in args.plugins:
        plugins.append(LinearProbingPlugin(
            benchmark=benchmark_class(
                n_experiences=1,
                shuffle=True,
                seed=seed,
                train_transform=train_transform,
            ),
            num_classes=num_classes
        ))

    # CREATE THE STRATEGY INSTANCE 
    if args.strategy == "naive":
        if args.loss_type == "self_supervised":
            cl_strategy = SelfSupervisedNaive(
                model,
                optimizer,
                BarlowTwinsLoss(),
                ss_augmentations=BTTrainingAugmentations(image_size=args.image_size),
                train_mb_size=args.batch_size,
                train_epochs=args.epochs,
                eval_mb_size=args.batch_size,
                device=device,
                plugins=plugins,
                evaluator=eval_plugin,
            )
        elif args.loss_type == "supervised":
            cl_strategy = Naive(
                model,
                optimizer,
                CrossEntropyLoss(),
                train_mb_size=args.batch_size,
                train_epochs=args.epochs,
                eval_mb_size=args.batch_size,
                device=device,
                plugins=plugins,
                evaluator=eval_plugin,
            )
    elif args.strategy == "cumulative":
        cl_strategy = Cumulative(
            model,
            optimizer,
            CrossEntropyLoss(),
            train_mb_size=args.batch_size,
            train_epochs=args.epochs,
            eval_mb_size=args.batch_size,
            device=device,
            plugins=plugins,
            evaluator=eval_plugin,
        )
    elif args.strategy == "lwf":
        cl_strategy = LwF(
            model,
            optimizer,
            CrossEntropyLoss(),
            alpha=args.alpha,
            temperature=args.temperature,
            train_mb_size=args.batch_size,
            train_epochs=args.epochs,
            eval_mb_size=args.batch_size,
            device=device,
            plugins=plugins,
            evaluator=eval_plugin,
        )
    # ADD YOUR CUSTOM STRATEGIES HERE
    else:
        raise NotImplementedError
    
    if args.resume_from_checkpoint:
        cl_strategy, initial_exp = maybe_load_checkpoint(cl_strategy, checkpoint_path)
    else:
        initial_exp = 0

    # TRAINING LOOP
    print("Starting experiment...")
    
    for experience in benchmark.train_stream[initial_exp:]:
        print("Start of experience ", experience.current_experience)
        cl_strategy.train(experience)
        print("Training completed")

        print("Evaluation of the current strategy:")
        cl_strategy.eval(benchmark.test_stream)
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