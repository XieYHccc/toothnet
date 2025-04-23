import argparse
import os
import numpy as np
import torch

from tqdm import tqdm
from toothnet.generator import DentalModelGenerator, collate_fn

def parse_args():
    parser = argparse.ArgumentParser(description='Model Test')
    parser.add_argument('--input_dir_path', default="data/preprocessed_data_16000_samples", type=str)
    parser.add_argument('--split_txt_path', default="data/preprocessed_data_16000_samples/test_list.txt", type=str, help="split txt path.")
    parser.add_argument('--model_name', type=str, default="pointnet2")
    parser.add_argument('--checkpoint_path', default="saved/ckpts/pointnet2_test_val.h5", type=str)
    parser.add_argument('--checkpoint_path_bdl', default="ckpts/tgnet_bdl", type=str)
    return parser.parse_args()

def cal_metric(gt_labels, pred_sem_labels, pred_ins_labels, is_half=None):
    ins_label_names = np.unique(pred_ins_labels)
    ins_label_names = ins_label_names[ins_label_names != 0]
    IOU = 0
    F1 = 0
    ACC = 0
    SEM_ACC = 0
    IOU_arr = []
    for ins_label_name in ins_label_names:
        #instance iou
        ins_label_name = int(ins_label_name)
        ins_mask = pred_ins_labels==ins_label_name
        gt_label_uniqs, gt_label_counts = np.unique(gt_labels[ins_mask], return_counts=True)
        gt_label_name = gt_label_uniqs[np.argmax(gt_label_counts)]
        gt_mask = gt_labels == gt_label_name

        TP = np.count_nonzero(gt_mask * ins_mask)
        FN = np.count_nonzero(gt_mask * np.invert(ins_mask))
        FP = np.count_nonzero(np.invert(gt_mask) * ins_mask)
        TN = np.count_nonzero(np.invert(gt_mask) * np.invert(ins_mask))

        ACC += (TP + TN) / (FP + TP + FN + TN)
        precision = TP / (TP+FP)
        recall = TP / (TP+FN)
        F1 += 2*(precision*recall) / (precision + recall)
        IOU += TP / (FP+TP+FN)
        IOU_arr.append(TP / (FP+TP+FN))
        #segmentation accuracy
        pred_sem_label_uniqs, pred_sem_label_counts = np.unique(pred_sem_labels[ins_mask], return_counts=True)
        sem_label_name = pred_sem_label_uniqs[np.argmax(pred_sem_label_counts)]
        if is_half:
            if sem_label_name == gt_label_name or sem_label_name + 8 == gt_label_name:
                SEM_ACC +=1
        else:
            if sem_label_name == gt_label_name:
                SEM_ACC +=1
        #print("gt is", gt_label_name, "pred is", sem_label_name, sem_label_name == gt_label_name)
    return IOU/len(ins_label_names), F1/len(ins_label_names), ACC/len(ins_label_names), SEM_ACC/len(ins_label_names), IOU_arr

def main(args):
    # load the model
    model = None
    if args.model_name == "pointnet2":
        from toothnet.models.pointnet2 import PointNet2
        model = PointNet2({})
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.cuda()
    model.eval()

    # load test data set
    generator = DentalModelGenerator(args.input_dir_path, args.split_txt_path, None)
    test_dataset = torch.utils.data.DataLoader(generator, batch_size=1, shuffle=False, collate_fn=collate_fn)
    print("The number of test data is: %d" % len(test_dataset))

    total_iou = 0
    total_f1 = 0
    total_acc = 0
    total_sem_acc = 0
    count = 0
    with torch.no_grad():
        for i, batch_item in tqdm(enumerate(test_dataset), total=len(test_dataset), smoothing=0.9):
            gt_labels = batch_item["gt_seg_label"]
            low_feat = batch_item["feat"].cuda()
            outputs = model(low_feat)
            pred_labels = outputs["labels"].squeeze()
            iou, f1, acc, sem_acc, iou_arr = cal_metric(gt_labels.squeeze(), pred_labels, pred_labels)
            total_iou += iou
            total_f1 += f1
            total_acc += acc
            total_sem_acc += sem_acc
            count += 1

    print(f"Average IOU: {total_iou / count:.4f}")
    print(f"Average F1: {total_f1 / count:.4f}")
    print(f"Average ACC: {total_acc / count:.4f}")
    print(f"Average SEM ACC: {total_sem_acc / count:.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)











