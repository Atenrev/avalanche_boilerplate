import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--loss_type", type=str, default="self_supervised",
                        choices=["supervised", "self_supervised"],
                        help="Type of loss to use for the benchmark")
    parser.add_argument("--strategy", type=str, default="naive",
                        choices=["naive", "cumulative", "lwf"], # ADD YOUR CUSTOM STRATEGIES HERE
                        help="Strategy to use for the benchmark")
    
    # ADD CUSTOM PARAMETERS HERE

    # Model parameters
    parser.add_argument("--model", type=str, default="resnet18_bt",
                        choices=["simple_mlp", "resnet18", "resnet18_bt"],
                        help="Model to use for the benchmark")

    # Benchmark parameters
    parser.add_argument("--benchmark", type=str, default="split_cifar10",
                        choices=["split_mnist", "split_fashion_mnist", "split_cifar10", "split_cifar100",
                                 "concon_disjoint", "concon_strict", "concon_unconf"],
                        help="Benchmark to use for the experiment")
    parser.add_argument("--eval_benchmarks", type=str, nargs='+', default=["split_cifar10"],
                        choices=["split_mnist", "split_fashion_mnist", "split_cifar10", "split_cifar100",
                                 "concon_disjoint", "concon_strict", "concon_unconfounded"],
                        help="Benchmarks to use for evaluation")
    parser.add_argument("--dataset_root", type=str, default="data/concon",
                        help="Root directory of the dataset")
    parser.add_argument("--n_experiences", type=int, default=1,
                        help="Number of experiences to use for the benchmark")
    parser.add_argument("--image_size", type=int, default=32,
                        help="Image size to use for the benchmark")
    parser.add_argument("--transform", type=str, default="barlow_twins",
                        choices=["none", "mnist", "cifar", "barlow_twins"],
                        help="Transform to use for the benchmark")
    parser.add_argument("--metrics", type=str, nargs='+', default=["loss", "accuracy", "forgetting"], 
                        choices=["loss", "accuracy", "forgetting", ],
                        help="Metrics to use for the benchmark")
    
    # Plugins
    parser.add_argument("--plugins", type=str, nargs='+', default=["linear_probing"],
                        choices=["linear_probing"],
                        help="Plugins to use for the benchmark")
    
    # General training parameters
    parser.add_argument("--epochs", type=int, default=1,
                        help="Number of epochs to use for the benchmark")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size to use for the benchmark")
    
    # Optimizer parameters
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate to use for the benchmark")
    
    # Linear Probing parameters
    parser.add_argument("--probe_epochs", type=int, default=4,
                        help="Number of epochs to use for the linear probing classifier")
    parser.add_argument("--probe_lr", type=float, default=1e-3,
                        help="Learning rate to use for the linear probing classifier")

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
    parser.add_argument("--wandb", action="store_true", default=True,
                        help="Use wandb for logging")
    
    return parser.parse_args()
