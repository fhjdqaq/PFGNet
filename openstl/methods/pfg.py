import torch
from openstl.models import PFG_Model
from .base_method import Base_method


class PFG(Base_method):

    def __init__(self, **args):
        super().__init__(**args)

    def _build_model(self, **args):
        return PFG_Model(**args)

    def forward(self, batch_x, batch_y=None, **kwargs):
        pre_seq_length, aft_seq_length = self.hparams.pre_seq_length, self.hparams.aft_seq_length
        if aft_seq_length == pre_seq_length:
            pred_y = self.model(batch_x)
        elif aft_seq_length < pre_seq_length:
            pred_y = self.model(batch_x)
            pred_y = pred_y[:, :aft_seq_length]
        elif aft_seq_length > pre_seq_length:
            pred_y = []
            d = aft_seq_length // pre_seq_length
            m = aft_seq_length % pre_seq_length
            
            cur_seq = batch_x.clone()
            for _ in range(d):
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq)

            if m != 0:
                cur_seq = self.model(cur_seq)
                pred_y.append(cur_seq[:, :m])
            
            pred_y = torch.cat(pred_y, dim=1)
        return pred_y
    

    def training_step(self, batch, batch_idx):
        batch_x, batch_y = batch
        pred_y = self(batch_x)
        loss = self.criterion(pred_y, batch_y)

        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True,sync_dist=True)

        self.train_loss_epoch_metric.update(loss.detach())
        return loss

    def on_train_epoch_end(self):
        train_avg = self.train_loss_epoch_metric.compute()
        self.log('train_loss_epoch', train_avg, on_step=False, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        self.train_loss_epoch_metric.reset()

