import argparse
from .defaults.sampler_defaults import HparamsBianryLatent, add_sampler_args
from .defaults.binarygan_default import HparamsBinaryAE, add_vqgan_args
from .defaults.experiment_defaults import add_PRDC_args, add_sampler_FID_args, add_big_sample_args


# args for training of all models: dataset, EMA and loading
def add_training_args(parser):
    parser.add_argument("--amp", const=True, action="store_const", default=False)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--custom_dataset_path", type=str)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--ema_beta", type=float, default=0.995)
    parser.add_argument("--ema", const=True, action="store_const", default=False)
    parser.add_argument("--load_dir", type=str, default="test")
    parser.add_argument("--load_optim", const=True, action="store_const", default=False)
    parser.add_argument("--load_step", type=int, default=0)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--steps_per_update_ema", type=int, default=10)
    parser.add_argument("--train_steps", type=int, default=100000000)


# args required for logging
def add_logging_args(parser):
    parser.add_argument("--log_dir", type=str, default="test")
    parser.add_argument("--save_individually", const=True, action="store_const", default=False)
    parser.add_argument("--steps_per_checkpoint", type=int, default=25000)
    parser.add_argument("--steps_per_display_output", type=int, default=5000)
    parser.add_argument("--steps_per_eval", type=int, default=0)
    parser.add_argument("--steps_per_log", type=int, default=10)
    parser.add_argument("--steps_per_save_output", type=int, default=5000)
    parser.add_argument("--visdom_port", type=int, default=8097)
    parser.add_argument("--visdom_server", type=str)


def add_distributed_args(parser):
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')


def set_up_base_parser(parser):
    add_training_args(parser)
    add_logging_args(parser)
    add_distributed_args(parser)


def apply_parser_values_to_H(H, args):
    # NOTE default args in H will be overwritten by any default parser args
    args = args.__dict__
    for arg in args:
        if args[arg] is not None:
            H[arg] = args[arg]

    return H

def get_vqgan_parser_args(parser):
    add_vqgan_args(parser)
    parser_args = parser.parse_args()
    
    return parser_args

def args2H(args):
    H = HparamsBinaryAE(args.dataset)
    H = apply_parser_values_to_H(H, args)

    if not H.lr:
        H.lr = H.base_lr * H.batch_size

    return H

def get_vqgan_hparams(parser):
    # parser = argparse.ArgumentParser("Parser for setting up VQGAN training :)")
    # set_up_base_parser(parser)
    add_vqgan_args(parser)
    parser_args = parser.parse_args()
    H = HparamsBinaryAE(parser_args.dataset)
    H = apply_parser_values_to_H(H, parser_args)

    if not H.lr:
        H.lr = H.base_lr * H.batch_size

    return H


def get_sampler_H_from_parser(parser):
    parser_args = parser.parse_args()
    dataset = parser_args.dataset

    # has to be in this order to overwrite duplicate defaults such as batch_size and lr
    H = HparamsBinaryAE(dataset)
    H.vqgan_batch_size = H.batch_size  # used for generating samples and latents

    if parser_args.sampler == "bld":
        H_sampler = HparamsBianryLatent(dataset)
    else:
        raise NotImplementedError
    H.update(H_sampler)  # overwrites old (vqgan) H.batch_size
    H = apply_parser_values_to_H(H, parser_args)
    return H


def set_up_sampler_parser(parser):
    set_up_base_parser(parser)
    add_vqgan_args(parser)
    add_sampler_args(parser)
    return parser


def get_sampler_hparams():
    parser = argparse.ArgumentParser("Parser for training discrete latent sampler models :)")
    set_up_sampler_parser(parser)
    H = get_sampler_H_from_parser(parser)
    return H


def get_PRDC_hparams():
    parser = argparse.ArgumentParser("Script for calculating PRDC on trained samplers")
    add_PRDC_args(parser)
    parser = set_up_sampler_parser(parser)
    H = get_sampler_H_from_parser(parser)
    return H


def get_sampler_FID_hparams():
    parser = argparse.ArgumentParser("Script for calculating FID on trained samplers")
    add_sampler_FID_args(parser)
    parser = set_up_sampler_parser(parser)
    H = get_sampler_H_from_parser(parser)
    return H


def get_big_samples_hparams():
    parser = argparse.ArgumentParser("Script for generating larger-than-training samples")
    add_big_sample_args(parser)
    parser = set_up_sampler_parser(parser)
    H = get_sampler_H_from_parser(parser)
    return H
