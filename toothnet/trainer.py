import copy
import os
import torch
import wandb
from torch.optim.lr_scheduler import ExponentialLR
from toothnet.external.scheduler import build_scheduler_from_cfg
from toothnet.loss_utils import LossMeter
from toothnet.generator import get_generator_set
from toothnet.loss_utils import LossMeter
from tqdm import tqdm
from math import inf

class Trainer:
    def __init__(self, config:dict, model):
        """
        Args:
            config: dict from get_train_configs() in train_config_maker.py
            model: nn.Module
        """

        self.config = config
        self.model = model(config)
        self.model.cuda()
        self.val_count = 0
        self.train_count = 0
        self.step_count = 0
        self.best_val_loss = inf
        self.train_loader, self.val_loader = get_generator_set(self.config["generator"], False)
        print(f"train set batch num: {len(self.train_loader)}")
        print(f"validation set batch num: {len(self.val_loader)}")

        if config["wandb"]["wandb_on"]:
            wandb.init(
                entity=self.config["wandb"]["entity"],
                project=self.config["wandb"]["project"],
                notes=self.config["wandb"]["notes"],
                tags=self.config["wandb"]["tags"],
                name=self.config["wandb"]["name"],
                config=self.config,
            )
        if self.config["tr_set"]["optimizer"]["NAME"] == "sgd":
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config["tr_set"]["optimizer"]["lr"],
                                             momentum=self.config["tr_set"]["optimizer"]["momentum"],
                                             weight_decay=self.config["tr_set"]["optimizer"]["weight_decay"])
        elif self.config["tr_set"]["optimizer"]["NAME"] == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["tr_set"]["optimizer"]["lr"],
                                              weight_decay=self.config["tr_set"]["optimizer"]["weight_decay"])

        if self.config["tr_set"]["scheduler"]["sched"] == "exp":
            self.scheduler = ExponentialLR(self.optimizer, self.config["tr_set"]["scheduler"]["step_decay"])
        elif self.config["tr_set"]["scheduler"]["sched"] == "cosine":
            sched_config = copy.deepcopy(self.config)
            sched_config["tr_set"]["scheduler"]["full_steps"] = self.config["tr_set"]["scheduler"]["full_steps"]
            sched_config["tr_set"]["scheduler"]["lr"] = self.config["tr_set"]["optimizer"]["lr"]
            sched_config["tr_set"]["scheduler"]["min_lr"] = self.config["tr_set"]["scheduler"]["min_lr"]
            self.scheduler = build_scheduler_from_cfg(sched_config["tr_set"]["scheduler"], self.optimizer)

    def save_model(self, phase):
        if not os.path.exists(os.path.dirname(self.config["checkpoint_path"])):
            os.makedirs(os.path.dirname(self.config["checkpoint_path"]), exist_ok=True)

        if phase == "train":
            torch.save(self.model.state_dict(), self.config["checkpoint_path"] + ".h5")
        elif phase == "val":
            torch.save(self.model.state_dict(), self.config["checkpoint_path"] + "_val.h5")
        else:
            raise "phase is something unknown"
        
    def train(self, epoch):
        print(f"--- Epoch {epoch} ---")
        self.model.train()
        total_loss_meter = LossMeter()
        step_loss_meter = LossMeter()
        pre_step = self.step_count
        for batch_idx, batch_item in tqdm(enumerate(self.train_loader), total=len(self.train_loader), smoothing=0.9):
            points = batch_item["feat"].cuda()
            seg_label = batch_item["gt_seg_label"].cuda()
            loss = self.model(points, seg_label)
            loss_sum = loss.get_sum()
            self.optimizer.zero_grad()
            loss_sum.backward()
            self.optimizer.step()
            torch.cuda.empty_cache()
            total_loss_meter.aggr(loss.get_loss_dict_for_print("train"))
            step_loss_meter.aggr(loss.get_loss_dict_for_print("step"))
            # print(loss.get_loss_dict_for_print("step"))
            if ((batch_idx + 1) % self.config["tr_set"]["scheduler"]["schedueler_step"] == 0) or (
                    self.step_count == pre_step and batch_idx == len(self.train_loader) - 1):
                if self.config["wandb"]["wandb_on"]:
                    wandb.log(step_loss_meter.get_avg_results(), step=self.step_count)
                    wandb.log({"step_lr": self.scheduler.get_last_lr()[0]}, step=self.step_count)
                self.step_count += 1
                self.scheduler.step(self.step_count)
                step_loss_meter.init()

        if self.config["wandb"]["wandb_on"]:
            wandb.log(total_loss_meter.get_avg_results(), step=self.step_count)
            self.train_count += 1
        self.save_model("train")

    def test(self, epoch, save_best_model):
        self.model.eval()
        total_loss_meter = LossMeter()
        with torch.no_grad():
            for batch_idx, batch_item in enumerate(self.val_loader):
                points = batch_item["feat"].cuda()
                seg_label = batch_item["gt_seg_label"].cuda()
                loss = self.model(points, seg_label)
                total_loss_meter.aggr(loss.get_loss_dict_for_print("val"))

        avg_total_loss = total_loss_meter.get_avg_results()
        if self.config["wandb"]["wandb_on"]:
            wandb.log(avg_total_loss, step=self.step_count)
            self.val_count += 1
        if save_best_model:
            if self.best_val_loss > avg_total_loss["total_val"]:
                self.best_val_loss = avg_total_loss["total_val"]
                self.save_model("val")

    def run(self):
        for epoch in range(60):
            self.train(epoch)
            self.test(epoch, True)