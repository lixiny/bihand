import sys
import pickle
import torch
import os
from torch.utils import data
from termcolor import colored, cprint
import bihand.utils.quatutils as quatutils
import bihand.config as cfg
import numpy as np
from bihand.losses.sikloss import SIKLoss

sik1m_inst = 0


class _SIK1M(data.Dataset):
    """
    The Loader for joints so3 and quat
    """

    def __init__(
            self,
            data_root="/disk1/data",
            data_source=None
    ):
        if data_source is None:
            data_source = ["sik-fh", 'sik-1m']
        data_path = os.path.join(data_root, 'SIK-1M')
        if not os.path.exists(data_path):
            raise ValueError("SIK-1M dataset: %s not exist" % data_path)

        print("Initialize _SIK1M instance")

        self.jointR = []
        self.quat = []
        self.ref_bone_link = (0, 9)  # mid mcp
        self.joint_root_idx = 9  # root

        for sik_sub in data_source:  # sik-fh, sik-1m
            pkl_path = os.path.join(data_path, '%s.pkl' % sik_sub)
            with open(pkl_path, 'rb') as fid:
                data = dict(pickle.load(fid))
                fid.close()
            jointR = data['joint_']
            quat = data['quat']

            self.jointR.append(jointR)
            self.quat.append(quat)
            cprint("SIK1M with source: {} init, total {}"
                   .format(sik_sub, jointR.shape[0]),
                   'magenta', attrs=['bold']
                   )

        self.jointR = np.concatenate(self.jointR, axis=0)
        self.quat = np.concatenate(self.quat, axis=0)

    def __len__(self):
        return len(self.jointR)

    def __getitem__(self, index):
        jointR = self.jointR[index]
        quat = self.quat[index]

        joint_bone = 0
        for j, nextj in zip(self.ref_bone_link[:-1], self.ref_bone_link[1:]):
            joint_bone += np.linalg.norm(jointR[nextj] - jointR[j])
        joint_bone = np.atleast_1d(joint_bone)
        jointRS = jointR / joint_bone

        kin_chain = [
            jointRS[i] - jointRS[cfg.SNAP_PARENT[i]]
            for i in range(21)
        ]
        kin_chain = np.array(kin_chain[1:])  # id 0's parent is itself
        kin_len = np.linalg.norm(
            kin_chain, ord=2, axis=-1, keepdims=True
        )
        kin_chain = kin_chain / kin_len

        jointRS = torch.from_numpy(jointRS).float()
        joint_bone = torch.from_numpy(joint_bone).float()
        kin_chain = torch.from_numpy(kin_chain).float()
        kin_len = torch.from_numpy(kin_len).float()
        quat = torch.from_numpy(quat).float()

        metas = {
            'jointRS': jointRS,
            'joint_bone': joint_bone,
            'kin_chain': kin_chain,
            'kin_len': kin_len,
            'quat': quat
        }
        return metas


class SIK1M(data.Dataset):
    def __init__(
            self,
            data_split="train",
            data_root="/disk1/data",
            split_ratio=0.8
    ):
        global sik1m_inst
        if not sik1m_inst:
            sik1m_inst = _SIK1M(data_root=data_root)
        self.sik1m = sik1m_inst
        self.permu = list(range(len(self.sik1m)))
        self.alllen = len(self.sik1m)
        self.data_split = data_split

        if data_split == "train":
            self.vislen = int(len(self.sik1m) * split_ratio)
            self.sub_permu = self.permu[:self.vislen]
        elif data_split in ["val", "test"]:
            self.vislen = self.alllen - int(len(self.sik1m) * split_ratio)
            self.sub_permu = self.permu[(self.alllen - self.vislen):]
        else:
            self.vislen = len(self.sik1m)
            self.sub_permu = self.permu[:self.vislen]

    def __len__(self):
        return self.vislen

    def __getitem__(self, index):
        return self.sik1m[self.sub_permu[index]]


def main():
    sik1m_train = SIK1M(
        data_split="train",
        data_root="/media/sirius/Lixin213G/Dataset"
    )
    sik1m_test = SIK1M(
        data_split="test"
    )

    metas = sik1m_train[2]
    for key in metas:
        print(key, metas[key].shape)

    quat = metas['quat']
    so3 = quatutils.quaternion_to_angle_axis(quat)


if __name__ == "__main__":
    main()
