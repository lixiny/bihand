# Copyright (c) Lixin YANG. All Rights Reserved.
r"""
Real world dataset
A Hand joint Tracking Benchmark from Stereo Matching, ICIP 2017
"""
import torch
import torch.utils.data
import os
import scipy.io as sio
import PIL
from PIL import Image
import numpy as np
import math
import pickle
from termcolor import colored, cprint
import bihand.utils.handutils as handutils
import bihand.config as cfg

CACHE_HOME = os.path.expanduser(cfg.DEFAULT_CACHE_DIR)

# some globals, ugly but work
sk_fx_color = 607.92271
sk_fy_color = 607.88192
sk_tx_color = 314.78337
sk_ty_color = 236.42484

bb_fx = 822.79041
bb_fy = 822.79041
bb_tx = 318.47345
bb_ty = 250.31296

sk_rot_vec = [0.00531, -0.01196, 0.00301]
sk_trans_vec = [-24.0381, -0.4563, -1.2326]  # mm

snap_joint_name2id = {w: i for i, w in enumerate(cfg.snap_joint_names)}
stb_joint_name2id = {w: i for i, w in enumerate(cfg.stb_joints)}

stb_to_snap_id = [snap_joint_name2id[joint_name] for joint_name in cfg.stb_joints]

def sk_rot_mx(rot_vec):
    """
    use Rodrigues' rotation formula to transform the rotation vector into rotation matrix
    :param rot_vec:
    :return:
    """
    theta = np.linalg.norm(rot_vec)
    vector = np.array(rot_vec) * math.sin(theta / 2.0) / theta
    a = math.cos(theta / 2.0)
    b = -vector[0]
    c = -vector[1]
    d = -vector[2]
    return np.array(
        [
            [
                a * a + b * b - c * c - d * d,
                2 * (b * c + a * d),
                2 * (b * d - a * c)
            ],
            [
                2 * (b * c - a * d),
                a * a + c * c - b * b - d * d,
                2 * (c * d + a * b)
            ],
            [
                2 * (b * d + a * c),
                2 * (c * d - a * b),
                a * a + d * d - b * b - c * c
            ]
        ]
    )

def sk_xyz_depth2color(depth_xyz, trans_vec, rot_mx):
    """
    in the STB dataset: 'rotation and translation vector can transform the coordinates
                         relative to color camera to those relative to depth camera'.
    however here we want depth_xyz -> color_xyz
    a inverse transformation happen:
    T = [rot_mx | trans_vec | 0  1], Tinv = T.inv, then output Tinv * depth_xyz

    :param depth_xyz: N x 21 x 3, trans_vec: 3, rot_mx: 3 x 3
    :return: color_xyz: N x 21 x 3
    """
    color_xyz = depth_xyz - np.tile(trans_vec, [depth_xyz.shape[0], depth_xyz.shape[1], 1])
    return color_xyz.dot(rot_mx)

def stb_palm2wrist(joint_xyz):
    root = snap_joint_name2id['loc_bn_palm_L'] # 0

    index = snap_joint_name2id['loc_bn_index_L_01'] # 5
    mid = snap_joint_name2id['loc_bn_mid_L_01']  # 9
    ring = snap_joint_name2id['loc_bn_ring_L_01']  # 13
    pinky = snap_joint_name2id['loc_bn_pinky_L_01'] #17

    def _new_root(joint_xyz, id, root_id):
        return joint_xyz[:, id, :] +\
            2.25 * (joint_xyz[:, root_id, :] - joint_xyz[:, id, :])  # N x K x 3

    joint_xyz[:, root, :] = \
        _new_root(joint_xyz, index, root) + \
        _new_root(joint_xyz, mid, root) + \
        _new_root(joint_xyz, ring, root) + \
        _new_root(joint_xyz, pinky, root)
    joint_xyz[:, root, :] = joint_xyz[:, root, :] / 4.0

    return joint_xyz

def _stb_palm2wrist(joint_xyz):
    root_id = snap_joint_name2id['loc_bn_palm_L']
    mid_root_id = snap_joint_name2id['loc_bn_mid_L_01']
    joint_xyz[:, root_id, :] =\
        joint_xyz[:, mid_root_id, :] + \
        2.2 * (joint_xyz[:, root_id, :] - joint_xyz[:, mid_root_id, :])  # N x K x 3
    return joint_xyz

class STBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root,
        data_split='train',
        hand_side='right',
        njoints=21,
        use_cache=True,
    ):
        if not os.path.exists(data_root):
            raise ValueError("data_root: %s not exist" % data_root)
        self.name        = 'stb'
        self.data_split  = data_split
        self.hand_side = hand_side
        self.img_paths   = []
        self.dep_paths   = []
        self.joints      = []
        self.kp2ds       = []
        self.centers     = []
        self.scales      = []
        self.njoints     = njoints # total 21 hand parts

        self.root_id     = snap_joint_name2id['loc_bn_palm_L']
        self.mid_mcp_id  = snap_joint_name2id['loc_bn_mid_L_01']
        ann_base         = os.path.join(data_root, "labels")
        img_base         = os.path.join(data_root, "images")
        sk_rot           = sk_rot_mx(sk_rot_vec)

        self.sk_intr = np.array([
            [sk_fx_color, 0.0, sk_tx_color],
            [0.0, sk_fy_color, sk_ty_color],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)  # (3,3)

        self.sequence = []
        if data_split == 'train':
            self.sequence = [
                "B2Counting",
                "B2Random",
                "B3Counting",
                "B3Random",
                "B4Counting",
                "B4Random",
                "B5Counting",
                "B5Random",
                "B6Counting",
                "B6Random"
            ]
        elif data_split == 'test':
            self.sequence = [
                "B1Counting",
                "B1Random"
            ]
        elif data_split == 'val':
            self.sequence = [
                "B2Counting",
                "B2Random"
            ]
        elif data_split == "train_val":
            self.sequence = [
                "B3Counting",
                "B3Random",
                "B4Counting",
                "B4Random",
                "B5Counting",
                "B5Random",
                "B6Counting",
                "B6Random"
            ]
        elif data_split == "all":
            self.sequence = [
                "B1Counting",
                "B1Random",
                "B2Counting",
                "B2Random",
                "B3Counting",
                "B3Random",
                "B4Counting",
                "B4Random",
                "B5Counting",
                "B5Random",
                "B6Counting",
                "B6Random"
            ]
        else:
            raise ValueError("split {} not in [train|test|val|train_val|all]")

        self.cache_folder = os.path.join(CACHE_HOME, "bihand-train", "stb")
        os.makedirs(self.cache_folder, exist_ok=True)
        cache_path = os.path.join(
            self.cache_folder, "{}.pkl".format(self.data_split)
        )
        if os.path.exists(cache_path) and use_cache:
            with open(cache_path, "rb") as fid:
                annotations = pickle.load(fid)
                self.img_paths = annotations["img_paths"]
                self.dep_paths = annotations["dep_paths"]
                self.joints = annotations["joints"]
                self.kp2ds = annotations["kp2ds"]
                self.centers = annotations["centers"]
                self.scales = annotations["scales"]
            print("stb {} gt loaded from {}".format(self.data_split, cache_path))
            return

        self.imgpath_list = [
            os.path.join(img_base, seq) for seq in self.sequence
        ]

        imgsk_prefix = "SK_color"
        depsk_prefix = "SK_depth_seg"

        annsk_list = [
            os.path.join(
                ann_base,
                "{}_{}.mat".format(seq, imgsk_prefix[:2])
            ) for seq in self.sequence
        ]

        self.ann_list = annsk_list

        for imgpath, ann in zip(self.imgpath_list, self.ann_list):
            ''' we only use SK image '''
            assert "SK" in ann
            ''' 1. load joint '''
            rawmat = sio.loadmat(ann)
            rawjoint = rawmat["handPara"].transpose((2,1,0)) # N x K x 3
            num = rawjoint.shape[0] # N

            rawjoint = sk_xyz_depth2color(rawjoint, sk_trans_vec, sk_rot)
            # reorder idx
            joint = rawjoint[:, stb_to_snap_id, :]
            # scale to meter
            joint = joint / 1000.0
            # root from palm to wrist
            joint = _stb_palm2wrist(joint)  # N x K x 3
            self.joints.append(joint)

            ''' 4. load images pth '''
            for idx in range(joint.shape[0]):
                self.img_paths.append(os.path.join(
                    imgpath, "{}_{}.png".format(imgsk_prefix, idx)
                ))
                self.dep_paths.append(os.path.join(
                    imgpath, "{}_{}.png".format(depsk_prefix, idx)
                ))

        self.joints = np.concatenate(self.joints, axis = 0).astype(np.float32)  ##(30000, 21, 3)

        for i in range(len(self.img_paths)):
            joint = self.joints[i]
            kp2d_homo = self.sk_intr.dot(joint.T).T
            kp2d = kp2d_homo / kp2d_homo[:, 2:3]
            kp2d = kp2d[:, :2]
            center = handutils.get_annot_center(kp2d)
            scale = handutils.get_annot_scale(kp2d)

            self.kp2ds.append( kp2d[np.newaxis,:,:] )
            self.centers.append( center[np.newaxis,:] )
            self.scales.append( (np.atleast_1d(scale))[np.newaxis,:] )

        self.kp2ds = np.concatenate(self.kp2ds, axis=0).astype(np.float32)  # (N, 21, 2)
        self.centers = np.concatenate(self.centers, axis=0).astype(np.float32)  # (N, 2)
        self.scales = np.concatenate(self.scales, axis=0).astype(np.float32)  # (N, 1)
        if use_cache:
            full_info = {
                "img_paths":self.img_paths,
                "dep_paths":self.dep_paths,
                "joints":self.joints,
                "kp2ds":self.kp2ds,
                "centers":self.centers,
                "scales":self.scales,
            }
            with open(cache_path, "wb") as fid:
                pickle.dump(full_info, fid)
                print("Wrote cache for dataset stb {} to {}".format(
                    self.data_split, cache_path
                ))
        return

    def __len__(self):
        """for STB dataset total (1,500 * 2) * 2 * 6 = 36,000 samples

        :return - if is train: 30,000 samples
        :return - if is eval:   6,000 samples
        """
        return len(self.img_paths)

    def __str__(self):
        info = "STB {} set. lenth {}".format(
            self.data_split, len(self.img_paths)
        )
        return colored(info, 'blue', attrs=['bold'])

    def _is_valid(self, clr, index):
        valid_data = isinstance(clr, (np.ndarray,PIL.Image.Image))

        if not valid_data:
            raise Exception("Encountered error processing stb[{}]".format(index))
        return valid_data

    def get_sample(self, index):  # replace __getitem__
        flip = True if self.hand_side != 'left' else False

        intr = self.sk_intr

        # prepare color image
        clr = Image.open(self.img_paths[index]).convert("RGB")
        self._is_valid(clr, index)

        # prepare depth image
        if self.dep_paths[index]:
            dep = Image.open(self.dep_paths[index]).convert("RGB")
            ### dep values now are stored as |mod|div|0| (RGB)
            self._is_valid(dep, index)
            valid_dep = True
        else:
            dep = None
            valid_dep = False

        # prepare joint
        joint = self.joints[index].copy()  #(21, 3)

        # prepare kp2d
        kp2d = self.kp2ds[index].copy()
        center = self.centers[index].copy()
        scale = self.scales[index].copy()

        if flip:
            clr = clr.transpose(Image.FLIP_LEFT_RIGHT)
            center[0] = clr.size[0] - center[0]
            kp2d[:, 0] = clr.size[0] - kp2d[:, 0]
            joint[:, 0] = -joint[:, 0]
            if valid_dep:
                dep = dep.transpose(Image.FLIP_LEFT_RIGHT)

        sample = {
            'index': index,
            'clr': clr,
            'dep': dep, # if has renturn PIL image
            'kp2d': kp2d,
            'center':center,
            'scale': scale,
            'joint': joint,
            'intr': intr,
            'valid_dep': valid_dep,
        }

        return sample

    def norm_dep_img(self, dep_, joint_z):
        if isinstance(dep_, PIL.Image.Image):
            dep_ = np.array(dep_)
            assert(dep_.shape[-1] == 3) # used as "RGB"

        ''' Converts a RGB-coded depth into float valued depth. '''
        ''' dep values now are stored as |mod|div|0| (RGB) '''
        dep = (dep_[:,:,1] * 2**8 + dep_[:,:,0]).astype('float32')
        dep /= 1000.0 # depth now in meter

        lower_bound = joint_z.min() - 0.05 # meter
        upper_bound = joint_z.max() + 0.05

        np.putmask(dep, dep<=lower_bound, upper_bound)
        min_dep = dep.min() - 1e-3  # slightly compensate
        np.putmask(dep, dep>=upper_bound, 0.0)
        max_dep = dep.max() + 1e-3
        np.putmask(dep, dep<=min_dep, max_dep)
        range_dep = max_dep - min_dep
        dep = (-1 * dep + max_dep)/range_dep
        return dep


def main():
    stb = STBDataset(
        data_root="/disk1/data/STB",
        data_split="train",
        hand_side="right"
    )
    sample = stb.get_sample(0)
    return sample

if __name__ == "__main__":
    main()
