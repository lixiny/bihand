import cv2
import numpy as np
import bihand.vis.renderer as renderer
import threading
import time
import bihand.utils.func as func
import bihand.utils.imgutils as imutils
import bihand.config as cfg
from queue import Queue
import os
from mano.webuser.smpl_handpca_wrapper_HAND_only import ready_arguments


class HandDrawer(threading.Thread):
    def __init__(self, reslu=512, mano_root=None):
        threading.Thread.__init__(self)
        if mano_root is None:
            mano_root = 'manopth/mano/models'
        mano_pth = os.path.join(mano_root, 'MANO_RIGHT.pkl')
        smpl_data = ready_arguments(mano_pth)
        faces = smpl_data['f'].astype(np.int32)

        self.reslu = reslu
        self.face = faces
        self.rend = renderer.MeshRenderer(self.face, img_size=reslu)
        self.exitFlag = 0
        self.drawingQueue = Queue(maxsize=20)
        self.fakeIntr = np.array([
            [reslu*2, 0, reslu/2],
            [0, reslu*2, reslu/2],
            [0, 0, 1]
        ])

    def run(self):
        backg = np.ones((self.reslu, self.reslu, 3)) * 255
        while(True):
            if self.exitFlag: break
            if self.drawingQueue.empty():
                time.sleep(0.1)
                continue
            drawing = self.drawingQueue.get()
            v = drawing['verts']
            clr = drawing['clr']
            uv = drawing['uv']
            resu1 = self.draw_verts(v, self.fakeIntr, backg)
            resu2 = self.draw_skeleton(clr, uv)
            demo = np.concatenate([resu1, resu2], axis=1)

            cv2.imshow('Rendered Hand', demo)
            cv2.waitKey(1)
        print("Hand Drawer finished")


    def feed(self, clr, verts, uv):
        clr = func.batch_denormalize(
            clr, [0.5, 0.5, 0.5], [1, 1, 1]
        )
        clr = func.bchw_2_bhwc(clr)
        clr = func.to_numpy(clr.detach().cpu())
        if clr.dtype is not np.uint8:
            clr = (clr * 255).astype(np.uint8)
        uv = func.to_numpy(uv)
        verts = func.to_numpy(verts)

        for i in range(clr.shape[0]): # batch_size
            drawing = {
                'verts': verts[i],
                'clr': clr[i],
                'uv': uv[i]
            }
            self.drawingQueue.put(drawing)

    def set_stop(self):
        self.exitFlag = 1


    def draw_skeleton(self, clr, uv):
        clr = imutils.draw_hand_skeloten(
            clr[...,::-1].copy(),  uv,  cfg.SNAP_BONES,  cfg.JOINT_COLORS
        )
        if clr.shape[0] != self.reslu:
            img = cv2.resize(clr, (self.reslu,self.reslu))
        return clr

    def draw_verts(self,verts, K, img):
        if img.shape[0] != self.reslu:
            img = cv2.resize(img, (self.reslu,self.reslu))
        resu = self.rend(verts, K ,img)
        resu = np.concatenate((resu[:,:,2:3],resu[:,:,1:2],resu[:,:,0:1]),2)
        return resu
