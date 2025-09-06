from fire import Fire
import os
import yaml
import json
from types import SimpleNamespace

import torch
from tensorboardX import SummaryWriter

import dataloader


def dict_to_namespace(d):
    """Convert a dictionary to a SimpleNamespace."""
    return json.loads(json.dumps(d), object_hook=lambda item: SimpleNamespace(**item))


def main(config_path: str = "config.yaml", **kwargs):
    # Load configuration from YAML file
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Update configuration with command line arguments
    # e.g. --training.batch_size=8
    for overriding_keys, overriding_value in kwargs.items():
        keys = overriding_keys.split('.')
        lowest_key = config_dict
        for key in keys[:-1]:
            lowest_key = config_dict[key]
        lowest_key[keys[-1]] = overriding_value

    args = dict_to_namespace(config_dict)

    gradient_accumulation_steps = args.training.gradient_accumulation_steps
    
    if gradient_accumulation_steps > 1:
        print('Effective Batch Size:', args.training.batch_size * gradient_accumulation_steps)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model_name = f"bs-{args.training.batch_size}"
    if args.training.gradient_accumulation_steps > 1:
        model_name += f"_acc-{gradient_accumulation_steps}"
    model_name += f"_lr-{args.training.learning_rate}"
    if args.training.use_scheduler:
        model_name += "_sched"
    model_name += f"_{args.experiment_name}"

    import datetime
    now = datetime.datetime.now()
    model_name += f"_{now.strftime('%Y-%m-%d_%H-%M-%S')}"
    
    print("Model name:", model_name)

    # Set up checkpoint and log directories
    if not os.path.exists(checkpoint_directory):
        os.makedirs(checkpoint_directory)
    log_directory = args.paths.log_directory
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    # Create subdirectories for the model
    checkpoint_directory = os.path.join(checkpoint_directory, model_name)
    train_writer = SummaryWriter(os.path.join(log_directory, model_name, "train"), max_queue=10, flush_secs=60)
    if args.training.validate:
        validate_writer = SummaryWriter(os.path.join(log_directory, model_name, "validate"), max_queue=10, flush_secs=60)

    resolution_scale = args.data.resolution_scale
    final_dimension = (int(224 * resolution_scale), int(400 * resolution_scale))
    print("Final dimension:", final_dimension)
    



if __name__ == "__main__":
    Fire(main)
