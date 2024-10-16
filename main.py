import os
import argparse
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

from src.benchmarks import (
    SplitMNISTBenchmark,
    SplitFashionMNISTBenchmark, 
    SplitCIFAR10Benchmark, 
    SplitCIFAR100Benchmark, 
)

from src.loggers import CSVLogger


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="naive",
                        choices=["naive", "cumulative", "lwf"], # ADD YOUR CUSTOM STRATEGIES HERE
                        help="Strategy to use for the benchmark")
    
    # ADD CUSTOM PARAMETERS HERE

    # Model parameters
    parser.add_argument("--model", type=str, default="simple_mlp",
                        choices=["simple_mlp", "resnet18"],
                        help="Model to use for the benchmark")

    # Benchmark parameters
    parser.add_argument("--benchmark", type=str, default="split_fashion_mnist",
                        choices=["split_mnist", "split_fashion_mnist", "split_cifar10", "split_cifar100"],
                        help="Benchmark to use for the experiment")
    parser.add_argument("--n_experiences", type=int, default=5,
                        help="Number of experiences to use for the benchmark")
    parser.add_argument("--image_size", type=int, default=32,
                        help="Image size to use for the benchmark")
    
    # General training parameters
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of epochs to use for the benchmark")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size to use for the benchmark")
    
    # Optimizer parameters
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate to use for the benchmark")

    # Learning without forgetting parameters
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Regularization factor for the distillation loss")
    parser.add_argument("--temperature", type=float, default=2.0,
                        help="Temperature for the distillation loss")

    # General experiment parameters
    parser.add_argument("--seeds", type=int, nargs='+', default=[1714],
                        help="Seed to use for the experiment. -1 to run the experiment with seeds 42, 69, 1714")
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    parser.add_argument("--resume_from_checkpoint", action="store_true", default=False,
                        help="Resume from a checkpoint")
    parser.add_argument("--remove_checkpoints", action="store_true", default=False,
                        help="Remove checkpoints after the experiment is completed")
    parser.add_argument("--output_dir", type=str,
                        default="results_debug/",
                        help="Output directory for the results")
    parser.add_argument("--project_name", type=str, default="AWESOME_CL_PROJECT",
                        help="Name of the project. If using wandb, this will be the name of the project")
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="Use wandb for logging")
    return parser.parse_args()


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
    else:
        raise NotImplementedError

    # CREATE THE OPTIMIZER
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)        

    # --- BENCHMARK CREATION      
    if args.benchmark == "split_mnist":
        benchmark = SplitMNISTBenchmark(
            n_experiences=args.n_experiences,
            shuffle=True,
            seed=seed,
        )  
    elif args.benchmark == "split_fashion_mnist":
        benchmark = SplitFashionMNISTBenchmark(
            n_experiences=args.n_experiences,
            shuffle=True,
            seed=seed,
        )
    elif args.benchmark == "split_cifar10":
        benchmark = SplitCIFAR10Benchmark(
            n_experiences=args.n_experiences,
            shuffle=True,
            seed=seed,
            image_size=args.image_size,
        )
    elif args.benchmark == "split_cifar100":
        benchmark = SplitCIFAR100Benchmark(
            n_experiences=args.n_experiences,
            shuffle=True,
            seed=seed,
            image_size=args.image_size,
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

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(minibatch=True, epoch=True, experience=True, trained_experience=True, stream=True),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True, stream=True),
        # ADD YOUR CUSTOM METRICS HERE
        loggers=loggers,
    )

    # CREATE THE PLUGINS
    plugins = []

    # CREATE THE STRATEGY INSTANCE 
    if args.strategy == "naive":
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
    args = __parse_args()
    main(args)