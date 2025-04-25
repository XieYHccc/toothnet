import argparse
import open3d.visualization as vis
import torch
from toothnet.predictor import Predictor
from toothnet.utils.o3d_utils import np_to_pcd
from toothnet.utils.io_utils import load_mesh, get_jaw_type_from_path
from toothnet.models.tsegnet import TSegNet

def parse_args():
    parser = argparse.ArgumentParser(description='Predict Unseen models')
    parser.add_argument('--mesh_path', default="data/preprocessed_data_16000_samples", type=str)
    parser.add_argument('--checkpoint_path', default="saved/ckpts/pointnet2_01_val.h5", type=str)
    parser.add_argument('--checkpoint_path_bdl', default="ckpts/tgnet_bdl", type=str)
    return parser.parse_args()

def main(args):
    model_config = {"run_tooth_segmentation_module" : False}
    model = TSegNet(model_config)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.cuda()
    model.eval()
    predictor = Predictor(model)

    vert_array = load_mesh(args.mesh_path)
    jaw_type = get_jaw_type_from_path(args.mesh_path)
    outputs = predictor.predict(vert_array=vert_array, jaw_type=jaw_type, output_names=["labels", "offset_result", "l3_xyz"])
    output_sampled_points = outputs["l3_xyz"]
    output_offsets = outputs["offset_result"]
    output_centroids = output_sampled_points + output_offsets

    o3d_pcd = np_to_pcd(vert_array[:, :3], [0.5, 0.5, 0])
    o3d_centroids = np_to_pcd(output_centroids, [0, 1, 0])
    vis.draw_geometries([o3d_pcd, o3d_centroids], mesh_show_wireframe = True, mesh_show_back_face = True)