# Copyright (c) Lixin YANG. All Rights Reserved.
r"""
Randered dataset
Learning to Estimate 3D Hand joint from Single RGB Images, ICCV 2017
"""
import torch
import torch.utils.data
import os
import PIL
from PIL import Image
import numpy as np
import pickle
from progress.progress.bar import Bar
from termcolor import colored, cprint
import bihand.utils.handutils as handutils
import bihand.config as cfg

CACHE_HOME = os.path.expanduser(cfg.DEFAULT_CACHE_DIR)

snap_joint_name2id = {w: i for i, w in enumerate(cfg.snap_joint_names)}
rhd_joint_name2id = {w: i for i, w in enumerate(cfg.rhd_joints)}
rhd_to_snap_id = [snap_joint_name2id[joint_name] for joint_name in cfg.rhd_joints]


class RHDDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data_root="/disk1/data/RHD/RHD_published_v2",
            data_split='train',
            hand_side='right',
            njoints=21,
            use_cache=True,
    ):

        if not os.path.exists(data_root):
            raise ValueError("data_root: %s not exist" % data_root)
        self.name = 'rhd'
        self.data_split = data_split
        self.hand_side = hand_side
        self.clr_paths = []
        self.dep_paths = []
        self.mask_paths = []
        self.joints = []
        self.kp2ds = []
        self.centers = []
        self.scales = []
        self.sides = []
        self.intrs = []
        self.njoints = njoints  # total 21 hand parts
        self.reslu = [320, 320]

        self.root_id = snap_joint_name2id['loc_bn_palm_L']  # 0
        self.mid_mcp_id = snap_joint_name2id['loc_bn_mid_L_01']  # 9

        # [train|test|val|train_val|all]
        if data_split == 'train':
            self.sequence = ['training', ]
        elif data_split == 'test':
            self.sequence = ['evaluation', ]
        elif data_split == 'val':
            self.sequence = ['evaluation', ]
        elif data_split == 'train_val':
            self.sequence = ['training', ]
        elif data_split == 'all':
            self.sequence = ['training', 'evaluation']
        else:
            raise ValueError("split {} not in [train|test|val|train_val|all]".format(data_split))

        self.cache_folder = os.path.join(CACHE_HOME, "bihand-train", "rhd")
        os.makedirs(self.cache_folder, exist_ok=True)
        cache_path = os.path.join(
            self.cache_folder, "{}.pkl".format(self.data_split)
        )
        if os.path.exists(cache_path) and use_cache:
            with open(cache_path, "rb") as fid:
                annotations = pickle.load(fid)
                self.sides = annotations["sides"]
                self.clr_paths = annotations["clr_paths"]
                self.dep_paths = annotations["dep_paths"]
                self.mask_paths = annotations["mask_paths"]
                self.joints = annotations["joints"]
                self.kp2ds = annotations["kp2ds"]
                self.intrs = annotations["intrs"]
                self.centers = annotations["centers"]
                self.scales = annotations["scales"]
            print("rhd {} gt loaded from {}".format(self.data_split, cache_path))
            return

        datapath_list = [
            os.path.join(data_root, seq) for seq in self.sequence
        ]
        annoname_list = [
            "anno_{}.pickle".format(seq) for seq in self.sequence
        ]
        anno_list = [
            os.path.join(datapath, annoname) \
            for datapath, annoname in zip(datapath_list, annoname_list)
        ]
        clr_root_list = [
            os.path.join(datapath, "color") for datapath in datapath_list
        ]
        dep_root_list = [
            os.path.join(datapath, "depth") for datapath in datapath_list
        ]
        mask_root_list = [
            os.path.join(datapath, "mask") for datapath in datapath_list
        ]

        print("init RHD {}, It will take a while at first time".format(data_split))
        for anno, clr_root, dep_root, mask_root \
                in zip(
            anno_list,
            clr_root_list,
            dep_root_list,
            mask_root_list
        ):

            with open(anno, 'rb') as fi:
                rawdatas = pickle.load(fi)
                fi.close()

            bar = Bar('RHD', max=len(rawdatas))
            for i in range(len(rawdatas)):
                raw = rawdatas[i]
                rawkp2d = raw['uv_vis'][:, : 2]  # kp 2d left & right hand
                rawvis = raw['uv_vis'][:, 2]
                rawjoint = raw['xyz']  # x, y, z coordinates of the keypoints, in meters
                rawintr = raw['K']
                ''' "both" means left, right'''
                kp2dboth = [
                    rawkp2d[:21][rhd_to_snap_id, :],
                    rawkp2d[21:][rhd_to_snap_id, :]
                ]
                visboth = [
                    rawvis[:21][rhd_to_snap_id],
                    rawvis[21:][rhd_to_snap_id]
                ]
                jointboth = [
                    rawjoint[:21][rhd_to_snap_id, :],
                    rawjoint[21:][rhd_to_snap_id, :]
                ]
                intrboth = [rawintr, rawintr]
                sideboth = ['l', 'r']

                maskpth = os.path.join(mask_root, '%.5d.png' % i)
                mask = Image.open(maskpth).convert("RGB")
                mask = np.array(mask)[:, :, 2:]
                id_left = [i for i in range(2, 18)]
                id_right = [i for i in range(18, 34)]
                np.putmask(mask, np.logical_and(mask >= id_left[0], mask <= id_left[-1]), 128)
                np.putmask(mask, np.logical_and(mask >= id_right[0], mask <= id_right[-1]), 255)
                area_left = np.sum(mask == 128)
                area_right = np.sum(mask == 255)
                vis_side = 'l' if area_left > area_right else 'r'

                for kp2d, vis, joint, side, intr \
                        in zip(kp2dboth, visboth, jointboth, sideboth, intrboth):
                    vis_sum = vis.sum()
                    if side != vis_side:
                        continue
                    clrpth = os.path.join(clr_root, '%.5d.png' % i)
                    deppth = os.path.join(dep_root, '%.5d.png' % i)
                    maskpth = os.path.join(mask_root, '%.5d.png' % i)
                    name = '%.5d' % i + side
                    self.clr_paths.append(clrpth)
                    self.dep_paths.append(deppth)
                    self.mask_paths.append(maskpth)
                    self.sides.append(side)

                    joint = joint[np.newaxis, :, :]
                    self.joints.append(joint)
                    center = handutils.get_annot_center(kp2d)
                    scale = handutils.get_annot_scale(kp2d)
                    kp2d = kp2d[np.newaxis, :, :]
                    self.kp2ds.append(kp2d)
                    center = center[np.newaxis, :]
                    self.centers.append(center)
                    scale = (np.atleast_1d(scale))[np.newaxis, :]
                    self.scales.append(scale)
                    intr = intr[np.newaxis, :]
                    self.intrs.append(intr)

                bar.suffix = ('({n}/{all}), total:{t:}s, eta:{eta:}s').format(
                    n=i + 1, all=len(rawdatas), t=bar.elapsed_td, eta=bar.eta_td)
                bar.next()
            bar.finish()
        self.joints = np.concatenate(self.joints, axis=0).astype(np.float32)  # (59629, 21, 3)
        self.kp2ds = np.concatenate(self.kp2ds, axis=0).astype(np.float32)  # (59629, 21, 2)
        self.centers = np.concatenate(self.centers, axis=0).astype(np.float32)  # (59629, 21, 2)
        self.scales = np.concatenate(self.scales, axis=0).astype(np.float32)  # (59629, 21, 1)
        self.intrs = np.concatenate(self.intrs, axis=0).astype(np.float32)  # (59629, 4)

        if use_cache:
            full_info = {
                "sides": self.sides,
                "clr_paths": self.clr_paths,
                "dep_paths": self.dep_paths,
                "mask_paths": self.mask_paths,
                "joints": self.joints,
                "kp2ds": self.kp2ds,
                "intrs": self.intrs,
                "centers": self.centers,
                "scales": self.scales,
            }
            with open(cache_path, "wb") as fid:
                pickle.dump(full_info, fid)
                print("Wrote cache for dataset rhd {} to {}".format(
                    self.data_split, cache_path
                ))
        return

    def get_sample(self, index):
        side = self.sides[index]
        """ 'r' in 'left' / 'l' in 'right' """
        flip = True if (side not in self.hand_side) else False
        valid_dep = True

        clr = Image.open(self.clr_paths[index]).convert("RGB")
        self._is_valid(clr, index)
        dep = Image.open(self.dep_paths[index]).convert("RGB")
        self._is_valid(dep, index)
        mask = Image.open(self.mask_paths[index]).convert("RGB")
        self._is_valid(mask, index)

        # prepare jont
        joint = self.joints[index].copy()

        # prepare kp2d
        kp2d = self.kp2ds[index].copy()

        center = self.centers[index].copy()
        scale = self.scales[index].copy()

        if flip:
            clr = clr.transpose(Image.FLIP_LEFT_RIGHT)
            dep = dep.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            center[0] = clr.size[0] - center[0]
            kp2d[:, 0] = clr.size[0] - kp2d[:, 0]
            joint[:, 0] = -joint[:, 0]

        dep = self._apply_mask(dep, mask, side)
        sample = {
            'index': index,
            'clr': clr,
            'dep': dep,
            'kp2d': kp2d,
            'center': center,
            'scale': scale,
            'joint': joint,
            'intr': self.intrs[index],
            'valid_dep': valid_dep,
        }

        return sample

    def _apply_mask(self, dep, mask, side):
        ''' follow the label rules in RHD datasets '''
        if side == 'l':
            valid_mask_id = [i for i in range(2, 18)]
        else:
            valid_mask_id = [i for i in range(18, 34)]

        mask = np.array(mask)[:, :, 2:]
        dep = np.array(dep)
        ll = valid_mask_id[0]
        uu = valid_mask_id[-1]
        mask[mask < ll] = 0
        mask[mask > uu] = 0
        mask[mask > 0] = 1
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        dep = np.multiply(dep, mask)
        dep = Image.fromarray(dep, mode="RGB")
        return dep

    def __len__(self):
        return len(self.clr_paths)

    def __str__(self):
        info = "RHD {} set. lenth {}".format(
            self.data_split, len(self.clr_paths)
        )
        return colored(info, 'yellow', attrs=['bold'])

    def norm_dep_img(self, dep_, joint_z):
        """RHD depthmap to depth image

        :param dm: depth map, RGB, R * 255 + G
        :type dm: np (H, W, 3)
        :param dm_mask: depth mask
        :type dm_mask: np (H, W, 3)
        :param hand_flag: 'l':left, 'r':right
        :type hand_flag: str
        :return: scaled dep image
        :rtype: np (H, W), a 0~1 float reptesent depth
        """
        if isinstance(dep_, PIL.Image.Image):
            dep_ = np.array(dep_)
            assert (dep_.shape[-1] == 3)  # used to be "RGB"

        ''' Converts a RGB-coded depth into float valued depth. '''
        dep = (dep_[:, :, 0] * 2 ** 8 + dep_[:, :, 1]).astype('float32')
        dep /= float(2 ** 16 - 1)
        dep *= 5.0  ## depth in meter !

        lower_bound = joint_z.min() - 0.05  # m
        upper_bound = joint_z.max() + 0.05

        np.putmask(dep, dep <= lower_bound, upper_bound)
        min_dep = dep.min() - 1e-3  # slightly compensate
        np.putmask(dep, dep >= upper_bound, 0.0)
        max_dep = dep.max() + 1e-3
        np.putmask(dep, dep <= min_dep, max_dep)
        range_dep = max_dep - min_dep
        dep = (-1 * dep + max_dep) / range_dep
        return dep

    def _is_valid(self, img, index):
        valid_data = isinstance(img, (np.ndarray, PIL.Image.Image))
        if not valid_data:
            raise Exception("Encountered error processing rhd[{}]".format(index))
        return valid_data


def main():
    rhd = RHDDataset(
        data_root="/disk2/data/RHD/RHD_published_v2",
        data_split='train',
        hand_side='right',
        njoints=21,
        use_cache=False
    )
    sample = rhd.get_sample(2145)

    # for name in sample:
    #     print("{} \n {} \n\n".format(name, sample[name]))


if __name__ == "__main__":
    main()
