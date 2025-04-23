import torch
import os
import numpy as np
import copy

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from toothnet.augmenter import Augmenter

class DentalModelGenerator(Dataset):
    def __init__(self, data_dir=None, split_with_txt_path=None, aug_obj_str=None):
        self.data_dir = data_dir
        # self.mesh_paths = glob(os.path.join(data_dir, "*_sampled_points.npy"))
        self.mesh_paths = []
        if split_with_txt_path:
            f = open(split_with_txt_path, 'r')
            while True:
                line = f.readline()
                if not line: break
                self.mesh_paths.append(os.path.join(data_dir,line.strip()))
            f.close()

        if aug_obj_str is not None:
            self.aug_obj = eval(aug_obj_str)
        else:
            self.aug_obj = None

    def __len__(self):
        return len(self.mesh_paths)

    def __getitem__(self, idx):
        mesh_arr = np.load(self.mesh_paths[idx].strip())
        output = {}

        low_feat = mesh_arr.copy()[:, :6].astype("float32")
        seg_label = mesh_arr.copy()[:, 6:].astype("int")
        seg_label -= 1  # -1 means gingiva, 0 means first incisor...

        if self.aug_obj:
            low_feat = self.aug_obj.augment(low_feat)

        low_feat = torch.from_numpy(low_feat)
        low_feat = low_feat.permute(1, 0)
        seg_label = torch.from_numpy(seg_label)
        seg_label = seg_label.permute(1, 0)

        output["feat"] = low_feat
        output["gt_seg_label"] = seg_label
        output["aug_obj"] = copy.deepcopy(self.aug_obj)
        output["mesh_path"] = self.mesh_paths[idx]
        return output


def collate_fn(batch):
    output = {}
    for batch_item in batch:
        for key in batch_item.keys():
            if key not in output:
                output[key] = []
            output[key].append(batch_item[key])

    for output_key in output.keys():
        if output_key in ["feat", "gt_seg_label", "uniform_feat", "uniform_gt_seg_label"]:
            output[output_key] = torch.stack(output[output_key])
    return output

def get_generator_set(config, is_test=False):
    if not is_test:
        point_loader = DataLoader(
            DentalModelGenerator(
                config["input_data_dir_path"],
                aug_obj_str=config["aug_obj_str"],
                split_with_txt_path=config["train_data_split_txt_path"]
            ),
            shuffle=True,
            batch_size=config["train_batch_size"],
            collate_fn=collate_fn,
            num_workers=10
        )

        val_point_loader = DataLoader(
            DentalModelGenerator(
                config["input_data_dir_path"],
                aug_obj_str=None,
                split_with_txt_path=config["val_data_split_txt_path"]

            ),
            shuffle=False,
            batch_size=config["val_batch_size"],
            collate_fn= collate_fn,
            num_workers=10
        )
        return [point_loader, val_point_loader]

# for test
if __name__ == "__main__":
    # data_generator = DentalModelGenerator("example_data/split_info/train_fold.txt", "aug.Augmentator([aug.Scaling([0.85, 1.15]), aug.Rotation([-30,30], 'fixed'), aug.Translation([-0.2, 0.2])])")
    data_generator = DentalModelGenerator("example_data/processed_data",
                                          "aug.Augmentator([aug.Flip(), aug.Scaling([0.85, 1.15]), aug.Rotation([-30,30], 'fixed'), aug.Translation([-0.2, 0.2])])")
    for batch in data_generator:
        for key in batch.keys():
            if type(batch[key]) == torch.Tensor:
                print(key, batch[key].shape)
            else:
                print(key, batch[key])
