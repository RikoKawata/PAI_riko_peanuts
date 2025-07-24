#
# Copyright (c) 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import os
import glob
import sys
import torch
import argparse
import numpy as np
import matplotlib.pylab as plt
import matplotlib.animation as anim
from eipl.utils import normalization
from eipl.utils import restore_args, tensor2numpy, deprocess_img

try:
    from libs.model_camLR import SARNN
except:
    sys.path.append("./libs/")
    from model_camLR_Hiera import SARNN

# argument parser
parser = argparse.ArgumentParser()
parser.add_argument("filename", type=str, default=None)
parser.add_argument("--idx", type=int, default=0)
args = parser.parse_args()

# restore parameters
dir_name = os.path.split(args.filename)[0]
# dir_name = './SARNN/log/20240722_1806_10'

params = restore_args(os.path.join(dir_name, "args.json"))
idx = args.idx

# load dataset
minmax = [params["vmin"], params["vmax"]]
joint_bound = np.load("../data/4direction_release_onlyLR/param/follower_joint_bound.npy")
leader_joint_bound = np.load("../data/4direction_release_onlyLR/param/leader_joint_bound.npy")
_joints = np.load("../data/4direction_release_onlyLR/test/follower_joint.npy")
_images_l = np.load("../data/4direction_release_onlyLR/test/left_img_128.npy")[:, :, ::2, ::2]
_images_l = np.transpose(_images_l, (0,1,4,2,3))
images_l = _images_l[idx]
_images_r = np.load("../data/4direction_release_onlyLR/test/right_img_128.npy")[:, :, ::2, ::2]
_images_r = np.transpose(_images_r, (0,1,4,2,3))
images_r = _images_r[idx]
joints = _joints[idx]


print("images left shape:{}, min={}, max={}".format(images_l.shape, images_l.min(), images_l.max()))
print("images right shape:{}, min={}, max={}".format(images_r.shape, images_r.min(), images_r.max()))
print("joints shape:{}, min={}, max={}".format(joints.shape, joints.min(), joints.max()))

# define model
model = SARNN(
    rec_dim=params["rec_dim"],
    union_dim = params["union_dim"],
    joint_dim=9,
    k_dim=params["k_dim"],
    heatmap_size=params["heatmap_size"],
    temperature=params["temperature"],
    im_size=[64,64]
)

# load weight
ckpt = torch.load(args.filename, map_location=torch.device("cpu"))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Inference

image_l_list,image_r_list, joint_list = [], [],[]
ect_pts_l_list, dec_pts_l_list = [], []
ect_pts_r_list, dec_pts_r_list = [], []
state = None
nloop = len(images_r)
# nloop = len(images_hand)

for loop_ct in range(nloop):
    # load data and normalization
    img_t = images_l[loop_ct]
    img_t = torch.Tensor(np.expand_dims(img_t, 0))
    img_t = normalization(img_t, (0, 255), (0.0,1.0))
    img_t_r = images_r[loop_ct]
    img_t_r = torch.Tensor(np.expand_dims(img_t_r, 0))
    img_t_r = normalization(img_t_r, (0, 255), (0.0,1.0))
    joint_t = torch.Tensor(np.expand_dims(joints[loop_ct], 0))
    joint_t = normalization(joint_t, joint_bound, minmax)
    
    # predict rnn
    y_image, y_image_r,y_joint, ect_pts, ect_pts_r, dec_pts,dec_pts_r, state = model(img_t,img_t_r, joint_t, state)

    # denormalization
    pred_image = tensor2numpy(y_image[0])
    pred_image = deprocess_img(pred_image,0.0,1.0)
    pred_image = pred_image.transpose(1, 2, 0)
    pred_image_r = tensor2numpy(y_image_r[0])
    pred_image_r = deprocess_img(pred_image_r, 0.0,1.0)
    pred_image_r = pred_image_r.transpose(1, 2, 0)
    pred_joint = tensor2numpy(y_joint[0])
    pred_joint = normalization(pred_joint, minmax, leader_joint_bound)

    # append data
    image_l_list.append(pred_image)
    image_r_list.append(pred_image_r)
    joint_list.append(pred_joint)
    ect_pts_l_list.append(tensor2numpy(ect_pts[0]))
    ect_pts_r_list.append(tensor2numpy(ect_pts_r[0]))
    dec_pts_l_list.append(tensor2numpy(dec_pts[0]))
    dec_pts_r_list.append(tensor2numpy(dec_pts_r[0]))

    #print("loop_ct:{}, joint:{}".format(loop_ct, pred_joint))
    #print("loop_ct:{}, joint:{}".format(loop_ct, joint_t))

pred_image = np.array(image_l_list)
pred_image_r = np.array(image_r_list)
pred_joint = np.array(joint_list)
# print(pred_joint[0])
# print(pred_joint[1])
# print(pred_joint[2])
# breakpoint()
def split_key_points(pts_list, img_size):
    pts = np.array(pts_list)
    pts = pts.reshape(-1, params["k_dim"], 2)
    pts[:, :, 0] = pts[:, :, 0] * img_size[0]
    pts[:, :, 1] = pts[:, :, 1] * img_size[1]
    pts[:, :, 0] = np.clip(pts[:, :, 0], 0, img_size[0])
    pts[:, :, 1] = np.clip(pts[:, :, 1], 0, img_size[1])
    return pts

ect_pts = split_key_points(ect_pts_l_list, img_size=(64,64))
dec_pts = split_key_points(dec_pts_l_list, img_size=(64,64))
ect_pts_hand = split_key_points(ect_pts_r_list, img_size=(64,64))
dec_pts_hand = split_key_points(dec_pts_r_list, img_size=(64,64))


# plot images
T = len(images_r)
# T_h = len(images_hand)
images = images_l.transpose(0,2,3,1)
images_hand = images_r.transpose(0,2,3,1)
fig, ax = plt.subplots(2, 3, figsize=(12, 5), dpi=60)

def anim_update(i):
    for j in range(2):
        for k in range(3):
            ax[j][k].cla()

    # plot camera image
    ax[0][0].imshow(images[i, :, :, ::-1])
    for j in range(params["k_dim"]):
        ax[0][0].plot(ect_pts[i, j, 0], ect_pts[i, j, 1], "bo", markersize=6)  # encoder
        ax[0][0].plot(
            dec_pts[i, j, 0], dec_pts[i, j, 1], "rx", markersize=6, markeredgewidth=2
        )  # decoder
    ax[0][0].axis("off")
    ax[0][0].set_title("Input left image")

    # plot predicted image
    ax[0][1].imshow(pred_image[i, :, :, ::-1])
    ax[0][1].axis("off")
    ax[0][1].set_title("Predicted image")

    # plot joint angle
    ax[0][2].set_ylim(0, 3200)
    ax[0][2].set_xlim(0, T)
    ax[0][2].plot(joints[1:], linestyle="dashed", c="k")
    # om has 5 joints, not 8
    for joint_idx in range(9):
        ax[0][2].plot(np.arange(i + 1), pred_joint[: i + 1, joint_idx])
    ax[0][2].set_xlabel("Step")
    ax[0][2].set_title("Joint angles")
    
    ax[1][0].imshow(images_hand[i, :, :, ::-1])
    for j in range(params["k_dim"]):
        ax[1][0].plot(ect_pts_hand[i, j, 0], ect_pts_hand[i, j, 1], "bo", markersize=6)  # encoder
        ax[1][0].plot(
            dec_pts_hand[i, j, 0], dec_pts_hand[i, j, 1], "rx", markersize=6, markeredgewidth=2
        )  # decoder
    ax[1][0].axis("off")
    ax[1][0].set_title("Input right image")

    # plot predicted image
    ax[1][1].imshow(pred_image_r[i, :, :, ::-1])
    ax[1][1].axis("off")
    ax[1][1].set_title("Predicted image")



ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(T / 10)), frames=T)
ani.save("./output/SARNN_camLR_{}_{}.gif".format(params["tag"], idx))

# If an error occurs in generating the gif animation, change the writer (imagemagick/ffmpeg).
# ani.save("./output/SARNN_{}_{}_{}.gif".format(params["tag"], idx, args.input_param), writer="ffmpeg")
