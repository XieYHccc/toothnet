import argparse
import os
import numpy as np
from glob import glob
from toothnet.io_utils import load_json, load_mesh
from toothnet.pointops_utils import pc_normalize, farthest_point_sample_torch

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_mesh_data_path',
                        default="/mnt/d/phd/dataset/data_obj_parent_directory", type=str,
                        help="data path in which original obj/stl data are saved")
    parser.add_argument('--source_json_data_path',
                        default="/mnt/d/phd/dataset/data_json_parent_directory", type=str,
                        help="data path in which original .json data are saved")
    parser.add_argument('--preprocessed_data_path', default="../preprocessed_data_16000_samples", type=str,
                        help="data path in which processed data will be saved")
    return parser.parse_args()


def main(args):
    os.makedirs(os.path.join(args.preprocessed_data_path), exist_ok=True)

    # get all the obj and stl files
    obj_path_ls = glob(os.path.join(args.source_mesh_data_path, "**/*.obj"), recursive=True)
    # stl_path_ls = glob(os.path.join(args.source_mesh_data_path, "**/*.obj"), recursive=True)

    # get all the json files
    json_path_map = {}
    for dir_path in [x[0] for x in os.walk(args.source_json_data_path)][1:]:
        for json_path in glob(os.path.join(dir_path, "*.json")):
            json_path_map[os.path.basename(json_path).split(".")[0]] = json_path

    # process
    for i in range(len(obj_path_ls)):
        print(f"processing: {i} {obj_path_ls[i]}")

        basename = os.path.basename(obj_path_ls[i]).split(".")[0]
        assert basename in json_path_map
        loaded_json = load_json(json_path_map[basename])
        labels = np.array(loaded_json['labels']).reshape(-1, 1)

        # map FDI labels to 0-16
        if loaded_json['jaw'] == 'lower':
            labels -= 20
        labels[labels // 10 == 1] %= 10
        labels[labels // 10 == 2] = (labels[labels // 10 == 2] % 10) + 8
        labels[labels < 0] = 0

        vert_array = load_mesh(obj_path_ls[i], process=False, maintain_order=True)
        if vert_array.shape[0] < 16000:
            print(f"mesh file {obj_path_ls[i]} only have {vert_array.shape[0]} vertices, dropped...")
            continue

        assert vert_array.shape[0] == labels.shape[0]

        # normalize the data
        vert_array[:, :3] = pc_normalize(vert_array[:, :3])
        # stds = np.std(pos_array, axis=0)
        # pos_array -= np.mean(pos_array, axis=0) / stds
        # pos_array = ((pos_array - pos_array.min(axis=0)) / (pos_array.max(axis=0) - pos_array.min(axis=0))) * 2 - 1

        labeled_vertices = np.concatenate([vert_array, labels], axis=1)

        # downsample the data
        samples_indexes = farthest_point_sample_torch(vert_array[:, :3], 16000)
        labeled_vertices = labeled_vertices[samples_indexes]

        np.save(os.path.join(args.preprocessed_data_path, f"{str(basename)}_sampled_points.npy"), labeled_vertices)


if __name__ == "__main__":
    args = parse_args()
    main(args)
