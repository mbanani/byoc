import pytorch_lightning as zeus
import torch
import pytorch3d

from ..models import build_model
from ..utils.metrics import evaluate_3d_correspondances, evaluate_pose_Rt, evaluate_feature_match
from ..utils.transformations import transform_points_Rt
from ..utils.util import detach_dictionary


class BYOC_Registration(zeus.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        # Build model
        self.model = build_model(cfg.model)
        self.num_views = cfg.dataset.num_views

        # Define weight losses
        self.loss_weights = {
            "vis": cfg.train.loss_weights.vis,
            "geo": cfg.train.loss_weights.geo,
            "v2g": cfg.train.loss_weights.v2g,
        }

    def batch_to_data(self, batch):
        if "rgb_0" in batch:
            rgb = [batch[f"rgb_{i}"].to(self.device) for i in range(self.num_views)]
            K = batch["K"].to(self.device)
        else:
            rgb = [None for i in range(self.num_views)]
            K = None

        pcs = [batch[f"points_{i}"].to(self.device) for i in range(self.num_views)]
        vps = [batch[f"Rt_{i}"].to(self.device) for i in range(self.num_views)]
        return rgb, pcs, vps, K

    def calculate_loss_and_metrics(self, batch, output, train=False):
        # evaluate losses and metrics
        loss, metrics = [], {}

        for f_type in ["vis", "geo", "v2g"]:
            metrics[f_type] = {}

            if f"{f_type}_corr_loss" in output:
                f_loss = self.loss_weights[f_type] * output[f"{f_type}_corr_loss"]
                loss.append(f_loss)
                metrics[f_type]["loss"] = f_loss

                if f_type != "v2g" and not train:
                    feat_metrics = self.evaluate_feature_type(batch, output, f_type)
                    metrics[f_type].update(feat_metrics)

        # loss is a list
        loss = sum(loss).mean()

        return loss, metrics

    def training_step(self, batch, batch_idx):
        return self.forward_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.forward_step(batch, batch_idx, "valid")

    def test_step(self, batch, batch_idx):
        return self.forward_step(batch, batch_idx, "test")

    def forward_step(self, batch, batch_idx, split):
        # forward pass
        rgb, pcs, gt_Rt, K = self.batch_to_data(batch)
        output = self.model(rgb, pcs, K)

        is_train = split == "train"
        loss, metrics = self.calculate_loss_and_metrics(batch, output, train=is_train)

        # Log everything
        self.log(f"loss/{split}", loss)
        for f_type in metrics:
            for metric in metrics[f_type]:
                val = metrics[f_type][metric].mean()
                self.log(f"{metric}_{f_type}/{split}", val)

        output = detach_dictionary(output)
        metrics = detach_dictionary(metrics)

        return {
            "loss": loss,
            "output": output,
            "metrics": metrics,
        }

    def evaluate_feature_type(self, batch, output, f_type):
        if f"{f_type}_corr" not in output:
            return {}

        rgb, pcs, gt_Rts, K = self.batch_to_data(batch)


        # Evaluate
        c_xyz_0, c_xyz_1, c_weight = output[f"{f_type}_corr"]
        gt_Rt = gt_Rts[1]
        pr_Rt = output[f"Rt_{f_type}"]

        pose_eval = evaluate_pose_Rt(pr_Rt, gt_Rt)

        if K is not None:
            _, _, H, W = rgb[0].shape
            corr_eval, _ = evaluate_3d_correspondances(c_xyz_0, c_xyz_1, K, gt_Rt, (H, W))
        else:
            corr_eval = {}

        # FMR Corr
        corr_0_r = transform_points_Rt(c_xyz_0, gt_Rt)
        dist_diff = (corr_0_r - c_xyz_1).norm(p=2, dim=2)
        fm_corr = (dist_diff < 0.1).float().mean(dim=1)
        fmr_corr = (fm_corr > 0.05).float()

        # FMR Unfiltered
        pc_0 = output[f"{f_type}_pc_0"]
        pc_1 = output[f"{f_type}_pc_1"]
        fm, fmr = evaluate_feature_match(
            pc_0, pc_1, gt_Rt, dist_thresh=0.1, inlier_thresh=0.05, num_sample=5000
        )

        return {
            **corr_eval,
            **pose_eval,
            "feature_match": fm,
            "FMR": fmr,
            "feature_match_corr": fm_corr,
            "FMR_Corr": fmr_corr,
        }

    def configure_optimizers(self):
        params = self.model.parameters()
        cfg = self.cfg.train
        optimizer = torch.optim.Adam(
            params, lr=cfg.lr, eps=1e-4, weight_decay=cfg.weight_decay
        )
        return optimizer

    def test_epoch_end(self, test_step_outputs):

        error_R = [t_o["metrics"]["geo"]["vp-error_R"] for t_o in test_step_outputs]
        error_t = [t_o["metrics"]["geo"]["vp-error_t"] for t_o in test_step_outputs]
        fmr_all = [t_o["metrics"]["geo"]["FMR"] for t_o in test_step_outputs]
        fmr_corr = [t_o["metrics"]["geo"]["FMR_Corr"] for t_o in test_step_outputs]

        error_R = torch.cat(error_R)
        error_t = torch.cat(error_t)
        fmr_all = torch.cat(fmr_all)
        fmr_corr = torch.cat(fmr_corr)


        recall_R = [(error_R <= thresh).float().mean() for thresh in [5, 10, 45]]
        recall_t = [(error_t <= thresh).float().mean() for thresh in [5, 10, 25]]

        print(
            "Pairwise Registration:   ",
            f"{recall_R[0] * 100:4.1f} ",
            f"{recall_R[1] * 100:4.1f} ",
            f"{recall_R[2] * 100:4.1f} ",
            f"{error_R.mean():4.1f} ",
            f"{error_R.median():4.1f} ",
            " || ",
            f"{recall_t[0] * 100:4.1f} ",
            f"{recall_t[1] * 100:4.1f} ",
            f"{recall_t[2] * 100:4.1f} ",
            f"{error_t.mean():4.1f} ",
            f"{error_t.median():4.1f} ",
        )
        print(
            "Feature Match Recall (Unfiltered):    ",
            f"{fmr_all.mean():.3f} ",
            f"{fmr_all.std() ** 2:.3f} ",
        )
        print(
            "Feature Match Recall (Filtered):      ",
            f"{fmr_corr.mean():.3f} ",
            f"{fmr_corr.std() ** 2:.3f} ",
        )
