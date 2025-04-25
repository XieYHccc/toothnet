import argparse
import open3d as o3d
import torch
from toothnet.predictor import Predictor
from toothnet.utils.o3d_utils import np_to_pcd_with_label
from toothnet.utils.io_utils import load_mesh, get_jaw_type_from_path

def parse_args():
    parser = argparse.ArgumentParser(description='Predict Unseen models')
    parser.add_argument('--mesh_path', default="data/05805yr_lower.obj", type=str)
    parser.add_argument('--model_name', type=str, default="pointnet2")
    parser.add_argument('--checkpoint_path', default="saved/ckpts/pointnet2_01_val.h5", type=str)
    parser.add_argument('--checkpoint_path_bdl', default="ckpts/tgnet_bdl", type=str)
    return parser.parse_args()

def main(args):
    predictor = None
    model = None
    if args.model_name == "pointnet2":
        from toothnet.models.pointnet2 import PointNet2
        model = PointNet2({})
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.cuda()
    model.eval()
    predictor = Predictor(model)

    vert_array = load_mesh(args.mesh_path)
    jaw_type = get_jaw_type_from_path(args.mesh_path)
    outputs = predictor.predict(vert_arr=vert_array, jaw_type=jaw_type, output_names=["labels"])
    o3d_pcd = np_to_pcd_with_label(vert_array[:, :3], outputs["labels"])
    o3d.visualization.draw_geometries([o3d_pcd], mesh_show_wireframe = True, mesh_show_back_face = True)

if __name__ == "__main__":
    args = parse_args()
    main(args)
