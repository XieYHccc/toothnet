import argparse
import torch
from configs import train_config_maker
from toothnet.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--model_name', default="tsegnet", type=str, help = "model name. list: tsegnet | tgnet_fps/tgnet_bdl | pointnet | pointnetpp | dgcnn | pointtransformer")
    parser.add_argument('--config_path', default="configs/tsegnet.py", type=str, help = "train config file path.")
    parser.add_argument('--experiment_name', default="tsegnet_centroid_seg_01", type=str, help = "experiment name.")
    parser.add_argument('--input_data_dir_path', default="data/preprocessed_data_16000_samples", type=str, help = "input data dir path.")
    parser.add_argument('--train_data_split_txt_path', default="data/preprocessed_data_16000_samples/train_list.txt", type=str, help = "train cases list file path.")
    parser.add_argument('--val_data_split_txt_path', default="data/preprocessed_data_16000_samples/val_list.txt", type=str, help = "val cases list file path.")
    return parser.parse_args()

def build_tsegnet_from_cfg(cfg):
    from toothnet.models.tsegnet import TSegNet
    model = TSegNet(cfg)
    if cfg.get("pretrained_centroid_model_path", None) is not None:
        model.load_state_dict(torch.load(cfg["pretrained_centroid_model_path"] +".h5"), strict=False)
    
    return model

def main(args):
    config = train_config_maker.get_train_config(
        args.config_path,
        args.experiment_name,
        args.input_data_dir_path,
        args.train_data_split_txt_path,
        args.val_data_split_txt_path,
    )

    trainer = None
    if args.model_name == "pointnet":
        from toothnet.models.pointnet import PointNet
        trainer = Trainer(config, PointNet)
    elif args.model_name == "pointnet2":
        from toothnet.models.pointnet2 import PointNet2
        trainer = Trainer(config=config, model=PointNet2)
    elif args.model_name =="tsegnet":
        trainer = Trainer(config=config, model=build_tsegnet_from_cfg(config))

    trainer.run()

if __name__ == "__main__":
    args = parse_args()
    main(args)

