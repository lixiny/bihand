import pickle
import torch
import os
from torch.utils import data
import bihand.config as cfg
import numpy as np
from termcolor import colored, cprint
from bihand.utils.eval.zimeval import EvalUtil
from progress.progress.bar import Bar


class SIKONLINE(data.Dataset):
    """
    The Loader for joints so3 and quat
    """
    def __init__(
        self,
        data_root="/disk1/data",
        data_split='train',
        data_source=None,
    ):
        self.impljointR = []
        self.targjointR = []
        self.ref_bone_link = (0, 9)  # mid mcp
        self.joint_root_idx = 9 # root

        data_path = os.path.join(data_root, 'SIK-online')
        if data_source is None:
            data_source = ['stb', 'rhd']

        stb_train = os.path.join(data_path, 'sik_train_stb_100epoch.pkl')
        rhd_train = os.path.join(data_path, 'sik_train_rhd_100epoch.pkl')

        stb_test = os.path.join(data_path, 'sik_test_stb.pkl')
        rhd_test = os.path.join(data_path, 'sik_test_rhd.pkl')

        if data_split == 'train':
            if 'stb' in data_source:
                with open(stb_train, 'rb') as fid:
                    raw = dict(pickle.load(fid))
                    self.impljointR.append(raw["jointImpl_"])
                    self.targjointR.append(raw["jointGt_"])
                    fid.close()
            if 'rhd' in data_source:
                with open(rhd_train, 'rb') as fid:
                    raw = dict(pickle.load(fid))
                    self.impljointR.append(raw["jointImpl_"])
                    self.targjointR.append(raw["jointGt_"])
                    fid.close()
        elif data_split in ['test', 'val']:
            if 'stb' in data_source:
                with open(stb_test, 'rb') as fid:
                    raw = dict(pickle.load(fid))
                    self.impljointR.append(raw["jointImpl_"])
                    self.targjointR.append(raw["jointGt_"])
                    fid.close()

            if 'rhd' in data_source:
                with open(rhd_test, 'rb') as fid:
                    raw = dict(pickle.load(fid))
                    self.impljointR.append(raw["jointImpl_"])
                    self.targjointR.append(raw["jointGt_"])
                    fid.close()

        self.impljointR = np.concatenate(self.impljointR, axis=0)
        self.targjointR = np.concatenate(self.targjointR, axis=0)
        cprint(
            'SIK-ONLINE {} set with source: {} init, total {}'
                .format(data_split, data_source, len(self.impljointR)),
            'blue', attrs=['bold']
        )

    def __len__(self):
        return len(self.impljointR)


    def _prepare_data(self, jointR):
        # 1.
        bone = 0
        for j, nextj in zip(self.ref_bone_link[:-1], self.ref_bone_link[1:]):
            bone += np.linalg.norm(jointR[nextj] - jointR[j])
        bone = np.atleast_1d(bone)
        # 2.
        jointRS = jointR / bone

        # 3,4.
        kin_chain = [
            jointRS[i] - jointRS[cfg.SNAP_PARENT[i]]
            for i in range(21)
        ]
        kin_chain = np.array(kin_chain[1:])  # id 0's parent is itself
        kin_len = np.linalg.norm(
            kin_chain, ord=2, axis=-1, keepdims=True
        )
        kin_chain = kin_chain / (kin_len + 1e-4)

        bone = torch.from_numpy(bone).float()
        jointRS = torch.from_numpy(jointRS).float()
        kin_chain = torch.from_numpy(kin_chain).float()
        kin_len = torch.from_numpy(kin_len).float()
        metas = {
            'joint_bone':bone, #1
            'jointRS':jointRS, #2
            'kin_chain':kin_chain, #3
            'kin_len':kin_len #4
        }
        return metas


    def __getitem__(self, index):
        impls = self._prepare_data(self.impljointR[index])
        targs = self._prepare_data(self.targjointR[index])
        return impls, targs

