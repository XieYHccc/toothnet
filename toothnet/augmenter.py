import numpy as np

def get_rotation_matrix(angle, axis):
    rad = np.deg2rad(angle)
    c, s = np.cos(rad), np.sin(rad)
    if axis == 'x':
        return np.array([[1, 0, 0],
                         [0, c, -s],
                         [0, s, c]])
    elif axis == 'y':
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])
    elif axis == 'z':
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")


class Augmenter:
    def __init__(self, translate = (-0.2, 0.2), scale = (0.8, 1.2), rotate = (-180, 180)):
        self.translate = translate
        self.scale = scale
        self.rotate = rotate

    def augment(self, vertex_arr):
        # start with identity 4x4 matrix
        transform = np.eye(4)

        # random rotation
        R = np.eye(3)
        if np.random.randint(0, 2):  # rotate X
            R = get_rotation_matrix(np.random.uniform(*self.rotate), 'x') @ R
        if np.random.randint(0, 2):  # rotate Y
            R = get_rotation_matrix(np.random.uniform(*self.rotate), 'y') @ R
        if np.random.randint(0, 2):  # rotate Z
            R = get_rotation_matrix(np.random.uniform(*self.rotate), 'z') @ R
        transform[:3, :3] = R @ transform[:3, :3]

        # random scaling
        if np.random.randint(0, 2):
            s = np.random.uniform(*self.scale)
            S = np.diag([s, s, s])
            transform[:3, :3] = transform[:3, :3] @ S

        vertex_arr[:, :3] = vertex_arr[:, :3] @ transform[:3, :3].T
        if vertex_arr.shape[1] == 6:
            vertex_arr[:, 3:6] = vertex_arr[:, 3:6] @ transform[:3, :3].T

        # random translation
        if np.random.randint(0, 2):
            T = [
                np.random.uniform(*self.translate),
                np.random.uniform(*self.translate),
                np.random.uniform(*self.translate)
            ]
            transform[:3, 3] = T

        vertex_arr[:, :3] += transform[:3, 3]
        return vertex_arr