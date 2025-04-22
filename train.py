import argparse
from configs import train_config_maker
from toothnet.trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--model_name', default="pointnet2", type=str, help = "model name. list: tsegnet | tgnet_fps/tgnet_bdl | pointnet | pointnetpp | dgcnn | pointtransformer")
    parser.add_argument('--config_path', default="configs/pointnet2.py", type=str, help = "train config file path.")
    parser.add_argument('--experiment_name', default="pointnet2_test", type=str, help = "experiment name.")
    parser.add_argument('--input_data_dir_path', default="data/preprocessed_data_16000_samples", type=str, help = "input data dir path.")
    parser.add_argument('--train_data_split_txt_path', default="data/preprocessed_data_16000_samples/train_list.txt", type=str, help = "train cases list file path.")
    parser.add_argument('--val_data_split_txt_path', default="data/preprocessed_data_16000_samples/val_list.txt", type=str, help = "val cases list file path.")
    return parser.parse_args()

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
        print("s")
    elif args.model_name == "pointnet2":
        from toothnet.models.pointnet2 import PointNet2
        trainer = Trainer(config=config, model=PointNet2)

    trainer.run()

if __name__ == "__main__":
    args = parse_args()
    main(args)

