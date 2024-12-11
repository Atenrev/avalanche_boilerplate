import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", type=str, default="naive",
                        choices=["naive", "cumulative"], # ADD YOUR CUSTOM STRATEGIES HERE
                        help="Strategy to use")
    
    # ADD CUSTOM PARAMETERS HERE

    # Model parameters
    parser.add_argument("--model", type=str, default="resnet_18",
                        choices=["simple_mlp", "resnet32s", "resnet_18", 
                                 "resnet18_encoder", "resnet18_mini_encoder"],
                        help="Model to use. Models that end with _encoder are used for self-supervised learning")

    # Benchmark parameters
    parser.add_argument("--benchmark", type=str, default="concon_strict",
                        choices=["split_mnist", "split_fashion_mnist", "split_cifar10", "split_cifar100",
                                 "concon_disjoint", "concon_strict", "concon_unconf"],
                        help="Benchmark to use for the experiment")
    parser.add_argument("--eval_benchmarks", type=str, nargs='+', default=["concon_strict", "concon_unconfounded"],
                        choices=["split_mnist", "split_fashion_mnist", "split_cifar10", "split_cifar100",
                                 "concon_disjoint", "concon_strict", "concon_unconfounded"],
                        help="Benchmarks to use for evaluation")
    parser.add_argument("--dataset_root", type=str, default="data/concon",
                        help="Root directory of the dataset")
    parser.add_argument("--n_experiences", type=int, default=1,
                        help="Number of experiences to use")
    parser.add_argument("--image_size", type=int, default=32,
                        help="Image size to use")
    parser.add_argument("--transform", type=str, default="cifar",
                        choices=["none", "mnist", "cifar", "barlow_twins", "emp_ssl"],
                        help="Transform to use")
    parser.add_argument("--metrics", type=str, nargs='+', default=["loss", "accuracy", "forgetting"], 
                        choices=["loss", "accuracy", "forgetting", ],
                        help="Metrics to use")
    
    # Plugins
    parser.add_argument("--plugins", type=str, nargs='+', default=["linear_probing"],
                        choices=["lwf", "ewc", "si", "linear_probing", "shrink_and_perturb"],
                        help="Plugins to use")
    
    # General training parameters
    parser.add_argument("--loss_type", type=str, default="supervised",
                        choices=["supervised", "self_supervised"],
                        help="Type of loss to use")
    parser.add_argument("--criterion", type=str, default="CE",
                        choices=["CE", "barlow_twins", "emp_ssl"],
                        help="Criterion to use for the training")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs to use")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size to use")
    parser.add_argument("--eval_every", type=int, default=-1,
                        help="Evaluate every n epochs. -1 to disable evaluation. 0 to evaluate only at the end of each experience")
    
    # Optimizer parameters
    parser.add_argument("--optimizer", type=str, default="sgd",
                        choices=["adam", "sgd", "lars"],
                        help="Optimizer to use")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="Learning rate to use with the optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay to use with the optimizer")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum to use with the optimizer")
    parser.add_argument("--nesterov", action="store_true", default=True,
                        help="Use Nesterov momentum with the optimizer")

    # Learning without forgetting parameters
    parser.add_argument("--lwf_alpha", type=float, default=1.0,
                        help="Regularization factor for the distillation loss")
    parser.add_argument("--lwf_temperature", type=float, default=2.0,
                        help="Temperature for the distillation loss")
    
    # Elastic Weight Consolidation parameters
    parser.add_argument("--ewc_lambda", type=float, default=1e-2,
                        help="Regularization factor for the EWC loss")
    
    # Synaptic Intelligence parameters
    parser.add_argument("--si_lambda", type=float, default=1e-2,
                        help="Regularization factor for the SI loss")
    
    # Linear Probing parameters
    parser.add_argument("--probe_epochs", type=int, default=100,
                        help="Number of epochs to use for the linear probing classifier")
    parser.add_argument("--probe_lr", type=float, default=1e-3,
                        help="Learning rate to use for the linear probing classifier")
    
    # Shrink and Perturb parameters
    parser.add_argument("--shrink", type=float, default=0.4,
                        help="Shrink factor to use")
    parser.add_argument("--perturb", type=float, default=0.1,
                        help="Perturb factor to use")
    parser.add_argument("--sp_every_epoch", action="store_true", default=False,
                        help="Shrink and perturb every epoch")
    parser.add_argument("--sp_every_experience", action="store_true", default=True,
                        help="Shrink and perturb every experience")

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
