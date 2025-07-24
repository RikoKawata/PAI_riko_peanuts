#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import torch
import torch.nn as nn
from eipl.utils import LossScheduler


class fullBPTTtrainer:
    """
    Helper class to train recurrent neural networks with numpy sequences

    Args:
        traindata (np.array): list of np.array. First diemension should be time steps
        model (torch.nn.Module): rnn model
        optimizer (torch.optim): optimizer
        input_param (float): input parameter of sequential generation. 1.0 means open mode.
    """

    def __init__(self, model, optimizer, loss_weights=[1.0, 1.0], device="cpu",laterality_ratio=1.0):
        self.device = device
        self.optimizer = optimizer
        self.loss_weights = loss_weights
        self.scheduler = LossScheduler(decay_end=1000, curve_name="s")
        self.model = model.to(self.device)
        self.laterality_ratio = laterality_ratio

    def save(self, epoch, loss, savename):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                #'optimizer_state_dict': self.optimizer.state_dict(),
                "train_loss": loss[0],
                "test_loss": loss[1],
            },
            savename,
        )

    def process_epoch(self, data, training=True):
        if not training:
            self.model.eval()
        else:
            self.model.train()

        total_loss = 0.0
        for n_batch, ((x_img_l, x_img_r,  x_joint), (y_img_l, y_img_r, y_joint)) in enumerate(data):
            if "cpu" in self.device:
                x_img_l = x_img_l.to(self.device)
                y_img_l = y_img_l.to(self.device)
                x_img_r = x_img_r.to(self.device)
                y_img_r = y_img_r.to(self.device)
                x_joint = x_joint.to(self.device)
                y_joint = y_joint.to(self.device)

            state = None
            yil_list, yir_list, yv_list = [], [],[]
            dec_pts_l_list, enc_pts_l_list = [], []
            dec_pts_r_list, enc_pts_r_list = [], []
            self.optimizer.zero_grad(set_to_none=True)
            for t in range(x_img_l.shape[1] - 1):
                _yil_hat, _yir_hat, _yv_hat, enc_l_ij,enc_r_ij, dec_l_ij,dec_r_ij, state = self.model(
                    x_img_l[:, t],x_img_r[:, t], x_joint[:, t], state
                )
                yil_list.append(_yil_hat)
                yir_list.append(_yir_hat)
                yv_list.append(_yv_hat)
                enc_pts_l_list.append(enc_l_ij)
                enc_pts_r_list.append(enc_r_ij)
                dec_pts_l_list.append(dec_l_ij)
                dec_pts_r_list.append(dec_r_ij)

            yil_hat = torch.permute(torch.stack(yil_list), (1, 0, 2, 3, 4))
            yir_hat = torch.permute(torch.stack(yir_list), (1, 0, 2, 3, 4))
            yv_hat = torch.permute(torch.stack(yv_list), (1, 0, 2))
            _enc_pts_l = torch.permute(torch.stack(enc_pts_l_list), (1, 0, 2))
            enc_pts_l = _enc_pts_l.reshape(_enc_pts_l.shape[0],_enc_pts_l.shape[1], -1,2)
            _enc_pts_r = torch.permute(torch.stack(enc_pts_r_list), (1, 0, 2))
            enc_pts_r = _enc_pts_r.reshape(_enc_pts_r.shape[0],_enc_pts_r.shape[1], -1,2)
            _dec_pts_l = torch.permute(torch.stack(dec_pts_l_list), (1, 0, 2))
            dec_pts_l = _dec_pts_l.reshape(_dec_pts_l.shape[0],_dec_pts_l.shape[1], -1,2)
            _dec_pts_r = torch.permute(torch.stack(dec_pts_r_list), (1, 0, 2))
            dec_pts_r = _dec_pts_r.reshape(_dec_pts_r.shape[0],_dec_pts_r.shape[1], -1,2)
            
            img_l_loss = nn.MSELoss()(yil_hat, y_img_l[:, 1:]) * self.loss_weights[0]
            img_r_loss = nn.MSELoss()(yir_hat, y_img_r[:, 1:]) * self.loss_weights[0]
            joint_loss = nn.MSELoss()(yv_hat, y_joint[:, 1:]) * self.loss_weights[1]
            # Gradually change the loss value using the LossScheluder class.
            pt_l_loss = nn.MSELoss()(
                torch.stack(dec_pts_l_list[:-1]), torch.stack(enc_pts_l_list[1:])
            ) * self.scheduler(self.loss_weights[2])
            pt_r_loss = nn.MSELoss()(
                torch.stack(dec_pts_r_list[:-1]), torch.stack(enc_pts_r_list[1:])
            ) * self.scheduler(self.loss_weights[2])
            pt_loss =  pt_l_loss + pt_r_loss
          
            enc_pt_x_loss_list , enc_pt_y_loss_list,dec_pt_x_loss_list,dec_pt_y_loss_list = [],[],[],[]
            for i in range(dec_pts_l.shape[2]):
                if (enc_pts_l[:,:,i,0].mean() > self.laterality_ratio) and (enc_pts_r[:,:,i,0].mean() < (1-self.laterality_ratio)):
                    enc_pt_x_loss = nn.MSELoss()(enc_pts_l[:,:,i,0], enc_pts_r[:,:,i,0])
                    dec_pt_x_loss = nn.MSELoss()(dec_pts_l[:,:,i,0], dec_pts_r[:,:,i,0])
                    enc_pt_y_loss = nn.MSELoss()(enc_pts_l[:,:,i,1], enc_pts_r[:,:,i,1])
                    dec_pt_y_loss = nn.MSELoss()(dec_pts_l[:,:,i,1], dec_pts_r[:,:,i,1])
                    enc_pt_x_loss_list.append(enc_pt_x_loss)
                    dec_pt_x_loss_list.append(dec_pt_x_loss)
                    enc_pt_y_loss_list.append(enc_pt_y_loss)
                    dec_pt_y_loss_list.append(dec_pt_y_loss)
            #attention
            stereo_ye_loss = torch.stack(enc_pt_y_loss_list).mean()* self.scheduler(self.loss_weights[3])
            stereo_yd_loss = torch.stack(dec_pt_y_loss_list).mean()* self.scheduler(self.loss_weights[3])
            stereo_xe_loss = torch.stack(enc_pt_x_loss_list).mean()* self.scheduler(self.loss_weights[3]*0.1)
            stereo_xd_loss = torch.stack(dec_pt_x_loss_list).mean()* self.scheduler(self.loss_weights[3]*0.1)
            stereo_loss = stereo_ye_loss +stereo_yd_loss +stereo_xe_loss +stereo_xd_loss

            # print(img_loss,joint_loss,pt_loss)
            loss =  img_l_loss + img_r_loss + joint_loss + pt_loss + stereo_loss
            # print("img",img_l_loss, "joint", joint_loss, "pt",pt_loss, stereo_loss)
            total_loss += loss.item()

            if training:
                
                loss.backward()
                self.optimizer.step()

        return total_loss / (n_batch + 1)
