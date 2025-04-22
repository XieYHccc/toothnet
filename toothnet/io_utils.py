import os
import json
import trimesh
import numpy as np

def load_json(file_path):
    with open(file_path, "r") as st_json:
        return json.load(st_json)

def load_txt(file_path):
    with open(file_path, "r") as st_txt:
        return [line.strip() for line in st_txt]
    
def load_mesh(file_path):
    mesh = trimesh.load_mesh(file_path, process=False, maintain_order=True)
    pos_array = np.array(mesh.vertices)
    normal_array = np.array(mesh.vertex_normals)
    concat = np.concatenate([pos_array, normal_array], axis=1)
    return concat

def get_jaw_type_from_path(path):
    list = os.path.basename(path).split('_')
    return list[1]

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
def save_predicted_labels(output_path, patient_id, jaw_type, labels):
    pred_output = {
        'id_patient': patient_id,
        'jaw': jaw_type,
        'labels': labels,
    }
    with open(output_path, 'w') as fp:
        json.dump(pred_output, fp, cls=NpEncoder)
    
