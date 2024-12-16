import os
import torch
import torch.optim.lr_scheduler

from types import MethodType

from torch import nn
from torch.nn import CrossEntropyLoss
from torchvision.models import resnet18

from avalanche.models import SimpleMLP, FeatureExtractorModel
from avalanche.models.resnet32 import ResNet, BasicBlock
from avalanche.training.determinism.rng_manager import RNGManager
from avalanche.checkpointing import maybe_load_checkpoint, save_checkpoint
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import (
    EvaluationPlugin, LwFPlugin, EWCPlugin, SynapticIntelligencePlugin, FeatureDistillationPlugin)
from avalanche.training.supervised import Naive, Cumulative
from avalanche.benchmarks import SplitMNIST, SplitFMNIST, SplitCIFAR10, SplitCIFAR100
from avalanche.training.self_supervised import Naive as SelfSupervisedNaive

from src.args import parse_args
from src.benchmarks import *
from src.optimizers import *
from src.transforms import *
from src.criterions import *
from src.loggers import *
from src.models import *
from src.plugins import *


def get_model(model_name, num_classes, device, no_head=False):
    if model_name == "simple_mlp":
        model = SimpleMLP(
            num_classes=num_classes,
            input_size=3 * args.image_size * args.image_size,
            hidden_size=512,
        )
        model.classifier = nn.Identity()

        if no_head:
            model = FeatureExtractorModel(
                model,
                nn.Identity(),
            )
        else:
            model = FeatureExtractorModel(
                model,
                nn.Linear(512, num_classes)
            )

    elif model_name == "resnet_18":
        model = resnet18(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Identity()

        if no_head:
            model = FeatureExtractorModel(
                model,
                nn.Identity(),
            )
        else:
            model = FeatureExtractorModel(
                model,
                nn.Linear(in_features, num_classes)
            )

    elif model_name == "resnet32s":
        model = ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)
        in_features = model.fc.in_features
        model.fc = nn.Identity()

        if no_head:
            model = FeatureExtractorModel(
                model,
                nn.Identity(),
            )
        else:
            model = FeatureExtractorModel(
                model,
                nn.Linear(in_features, num_classes)
            )

    elif model_name == "resnet18_encoder" or model_name == "resnet18_mini_encoder":
        model = ResNet18(mini_version=model_name == "resnet18_mini_encoder")
        model = FeatureExtractorModel(
            model,
            nn.Identity(),
        )

    else:
        raise NotImplementedError

    return model.to(device)


def get_benchmark(benchmark_name, seed, train_transform, eval_transform, n_experiences=1, dataset_root=None):
    base_params = {
        "n_experiences": n_experiences,
        "shuffle": True,
        "seed": seed,
        "train_transform": train_transform,
        "eval_transform": eval_transform,
    }

    if dataset_root is not None:
        base_params["dataset_root"] = dataset_root

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
    elif benchmark_name == "concon_disjoint":
        benchmark_class = ConConDisjoint
        num_classes = 2
    elif benchmark_name == "concon_unconfounded":
        benchmark_class = ConConUnconfounded
        num_classes = 2
    else:
        raise NotImplementedError

    benchmark = benchmark_class(**base_params)
    
    # Add the name of the benchmark at the beginning of the name of all streams
    benchmark.train_stream.name = f"{benchmark_name}_{benchmark.train_stream.name}"
    benchmark.test_stream.name = f"{benchmark_name}_{benchmark.test_stream.name}"
    
    # Update keys in stream_definitions
    if hasattr(benchmark, "stream_definitions"):
        benchmark.stream_definitions[f"{benchmark_name}_train"] = benchmark.stream_definitions.pop("train")
        benchmark.stream_definitions[f"{benchmark_name}_test"] = benchmark.stream_definitions.pop("test")
    
    # Update keys in streams
    benchmark._streams[f"{benchmark_name}_train"] = benchmark._streams.pop("train")
    benchmark._streams[f"{benchmark_name}_test"] = benchmark._streams.pop("test")
    
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
        "eval_every": args.eval_every,
    }

    if args.criterion == "CE":
        assert args.loss_type == "supervised"
        base_params["criterion"] = CrossEntropyLoss()
    elif args.criterion == "barlow_twins":
        assert args.loss_type == "self_supervised"
        base_params["criterion"] = BarlowTwinsLoss()
        base_params["ss_augmentations"] = BTTrainingAugmentations(
            image_size=args.image_size
        )
    elif args.criterion == "emp_ssl":
        assert args.loss_type == "self_supervised"
        base_params["criterion"] = EMPSLLLoss()
        base_params["ss_augmentations"] = EMPSSLTrainingAugmentations(
            image_size=args.image_size,
            num_patch=100
        )
    else:
        raise NotImplementedError

    if args.strategy == "naive":
        if args.loss_type == "self_supervised":
            strategy_class = SelfSupervisedNaive
            base_params["eval_criterion"] = torch.nn.CrossEntropyLoss()
        elif args.loss_type == "supervised":
            strategy_class = Naive
        else:
            raise NotImplementedError
    elif args.strategy == "cumulative":
        strategy_class = Cumulative
    # ADD YOUR CUSTOM STRATEGIES HERE
    else:
        raise NotImplementedError

    return strategy_class(**base_params)


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
        f"cuda:{args.cuda}" if torch.cuda.is_available(
        ) and args.cuda >= 0 else "cpu"
    )

    run_name = f"{args.strategy}_w_{args.model}_on_{args.benchmark}_loss({args.loss_type})_criterion({args.criterion})"
    run_name += f"_epochs({args.epochs})_exps({args.n_experiences})_lr({args.lr})_bs({args.batch_size})"

    # ADD CUSTOM PARAMETERS TO THE RUN NAME HERE

    if "lwf" in args.plugins:
        run_name += f"_lwf_alpha({args.lwf_alpha})_temp({args.lwf_temperature})"

    if "feature_distillation" in args.plugins:
        run_name += f"_fd_alpha({args.fd_alpha})_mode({args.fd_mode})"

    if "ewc" in args.plugins:
        run_name += f"_ewc_lambda({args.ewc_lambda})"

    if "si" in args.plugins:
        run_name += f"_si_lambda({args.si_lambda})"

    if "linear_probing" in args.plugins:
        run_name += "_lp"
        run_name += f"_lr({args.probe_lr})_epochs({args.probe_epochs})"

    if "shrink_and_perturb" in args.plugins:
        run_name += f"_shpe({args.shrink}_{args.perturb})"
        run_name += f"_every({args.sp_every})"
        
    if "random_perturb" in args.plugins:
        run_name += f"_rp_std({args.rp_std})_sensitivity({args.rp_sensitivity})"
        run_name += f"_every({args.rp_every})"

    if "vanilla_model_merging" in args.plugins:
        run_name += f"_vmm({args.merge_coeff})"
        run_name += f"_every({args.mm_every})"

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
    elif args.transform == "emp_ssl":
        train_transform, eval_transform = EMPSSLTransform(args.image_size)
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
    no_head = args.loss_type == "self_supervised"
    model = get_model(args.model, num_classes, device, no_head=no_head)

    # CREATE THE OPTIMIZER
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.lr)
    elif args.optimizer == "sgd" or args.optimizer == "lars":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            nesterov=args.nesterov,
            weight_decay=args.weight_decay
        )

        if args.optimizer == "lars":
            optimizer = LARSWrapper(
                optimizer, eta=0.005, clip=True, exclude_bias_n_norm=True,)
    else:
        raise NotImplementedError

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

    if "lwf" in args.plugins:
        plugins.append(LwFPlugin(
            alpha=args.lwf_alpha,
            temperature=args.lwf_temperature,
        ))

    if "feature_distillation" in args.plugins:
        plugins.append(FeatureDistillationPlugin(
            alpha=args.fd_alpha,
            mode=args.fd_mode,
        ))

    if "ewc" in args.plugins:
        plugins.append(EWCPlugin(
            ewc_lambda=args.ewc_lambda,
        ))

    if "si" in args.plugins:
        plugins.append(SynapticIntelligencePlugin(
            si_lambda=args.si_lambda,
        ))

    if "linear_probing" in args.plugins:
        plugins.append(LinearProbingPlugin(
            benchmark=get_benchmark(args.benchmark, seed, train_transform, eval_transform,
                                    n_experiences=1, dataset_root=args.dataset_root)[0],
            num_classes=num_classes,
            epochs=args.probe_epochs,
            lr=args.probe_lr,
        ))

    if "shrink_and_perturb" in args.plugins:
        plugins.append(ShrinkAndPerturbPlugin(
            shrink=args.shrink,
            perturb=args.perturb,
            every=args.sp_every
        ))
        
    if "random_perturb" in args.plugins:
        plugins.append(RandomPerturbPlugin(
            perturb_std_ratio=args.rp_std,
            magnitude_sensitivity=args.rp_sensitivity,
            every=args.rp_every
        ))

    if "vanilla_model_merging" in args.plugins:
        plugins.append(VanillaModelMergingPlugin(
            merge_coeff=args.merge_coeff,
            every=args.mm_every
        ))

    # CREATE THE STRATEGY INSTANCE
    cl_strategy = get_strategy(
        args, model, optimizer, device, plugins, eval_plugin)

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
