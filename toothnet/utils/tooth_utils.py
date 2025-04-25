import numpy as np

def map_fdi_to_flat_label(labels, jaw_type):
    """
    Map FDI dental numbering labels to flat indices in the range [0, 16), 
    suitable for model training or visualization. 0 is gingiva.

    Args:
        labels (np.ndarray): An array of FDI tooth labels (e.g., 11, 42, etc.).
        jaw_type (str): Either 'upper' or 'lower' to distinguish between arches.

    Returns:
        np.ndarray: An array of mapped labels in the range [0, 16),
                    where label 0 may also represent gingiva or background.
    """

    labels = labels.copy()
    if jaw_type == 'lower':
        labels -= 20
    labels[labels // 10 == 1] %= 10
    labels[labels // 10 == 2] = (labels[labels // 10 == 2] % 10) + 8
    labels[labels < 0] = 0

    return labels

