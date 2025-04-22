import importlib.util
import sys
import os

def get_default_train_config(experiment_name, input_data_dir_path, train_data_split_txt_path, val_data_split_txt_path):
    config = {
        # wandB options
        # if you dont want to use wandb, just set config["wandb"]["wandb_on"] value to False.
        "wandb":{
            "entity": "yihangxie20001013-unviersity-of-dundee", #change to your username
            "wandb_on": True,
            "project": "tooth mesh segmentation",
            "tags": "tooth mesh segmentation",
            "notes": "tooth mesh segmentation",
            "name": experiment_name,
        },
        # generator options
        # the fps sampled points are input to the model
        "generator":{
            "input_data_dir_path": f"{input_data_dir_path}",
            "train_data_split_txt_path": f"{train_data_split_txt_path}",
            "val_data_split_txt_path": f"{val_data_split_txt_path}",
            "aug_obj_str": "Augmenter(translate=(-0.2, 0.2), scale=(0.8, 1.2), rotate=(-180, 180))",
            "train_batch_size": 3,
            "val_batch_size": 3,
        },
        "checkpoint_path": f"saved/ckpts/{experiment_name}",
    }
    return config


def get_train_config(config_path, experiment_name, input_data_dir_path, train_data_split_txt_path, val_data_split_txt_path):
    config = {}
    config.update(get_default_train_config(experiment_name, input_data_dir_path, train_data_split_txt_path, val_data_split_txt_path))

    spec = importlib.util.spec_from_file_location("module.name", config_path)
    loaded_model_config = importlib.util.module_from_spec(spec)
    sys.modules["module.name"] = loaded_model_config
    spec.loader.exec_module(loaded_model_config)

    config.update(loaded_model_config.train_config)
    return config