import os
import sys
import cv2
import math
import random
import numpy as np
import time
import _pickle as cPickle
# from PIL import Image
from tqdm import tqdm
import torch.utils.data as data
import torch
from utils import load_depth, get_bbox
from mmengine import Registry
from .data_augmentation import defor_3D_pc, defor_3D_bb, defor_3D_rt, defor_3D_bc, deform_non_linear,get_rotation

DATASET_REGISTRY = Registry("DATASET")

@DATASET_REGISTRY.register_module()
class PoseDataset(data.Dataset):
    def __init__(self, source, mode, data_dir, n_pts, vis,img_size=192, per_obj=None,use_cache=False,use_augment=True):
        """
        Args:
            source: 'CAMERA', 'Real' or 'CAMERA+Real'
            mode: 'train' or 'test'
            data_dir:
            n_pts: number of selected foreground points
        """

        self.source = source
        self.mode = mode
        self.data_dir = data_dir
        self.n_pts = n_pts
        self.vis=vis
        self.per_obj=per_obj
        self.img_size=img_size
        self.use_augment=use_augment


        assert source in ['CAMERA', 'Real', 'CAMERA+Real']
        assert mode in ['train', 'test']
        img_list_path = ['CAMERA/train_list.txt', 'Real/train_list.txt',
                         'CAMERA/val_list.txt', 'Real/test_list.txt']
        model_file_path = ['obj_models/camera_train.pkl', 'obj_models/real_train.pkl',
                           'obj_models/camera_val.pkl', 'obj_models/real_test.pkl']
        if mode == 'train':
            del img_list_path[3:]
            del model_file_path[2:]
        else:
            del img_list_path[:3]
            del model_file_path[:2]
        if source == 'CAMERA':
            del img_list_path[-1]
            del img_list_path[-1]
            del model_file_path[-1]
        elif source == 'Real':
            del img_list_path[0]
            # del img_list_path[-1]
            del model_file_path[0]
        elif source=='CAMERA+Real':
            del img_list_path[2:]
        else:
            # only use Real to test when source is CAMERA+Real
            if mode == 'test':
                del img_list_path[0]
                del model_file_path[0]

        img_list = []
        subset_len = []
        for path in img_list_path:
            img_list += [os.path.join(path.split('/')[0], line.rstrip('\n'))
                         for line in open(os.path.join(data_dir, path))]
            subset_len.append(len(img_list))
        if len(subset_len) == 2:
            self.subset_len = [subset_len[0], subset_len[1]-subset_len[0]]

        if per_obj is not None:
            self.img_list=[]
            for img in img_list:
                img_path = os.path.join(self.data_dir, img)
                with open(img_path + '_label.pkl', 'rb') as f:
                    gts = cPickle.load(f)
                b=False
                for i in range(len(gts['instance_ids'])):
                    if gts['class_ids'][i]-1==self.per_obj:
                        b=True
                        break
                if b:
                    self.img_list.append(img)
        else:
            self.img_list = img_list
        self.length = len(self.img_list)


        self.random=list(range(self.length))
        random.seed(1002)
        random.shuffle(self.random)

        # meta info for re-label mug category
        with open(os.path.join(data_dir, 'obj_models/mug_meta.pkl'), 'rb') as f:
            self.mug_meta = cPickle.load(f)

        self.mean_shapes = np.load('assets/mean_points_emb.npy')
        self.cat_names = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']
        self.camera_intrinsics = [577.5, 577.5, 319.5, 239.5]    # [fx, fy, cx, cy]
        self.real_intrinsics = [591.0125, 590.16775, 322.525, 244.11084]
        self.sym_ids = [0, 1, 3]    # 0-indexed
        self.norm_scale = 1000.0    # normalization scale
        self.shift_range = 0.01

        models = {}
        for path in model_file_path:
            with open(os.path.join(data_dir, path), 'rb') as f:
                models.update(cPickle.load(f))
        self.models = models

        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])

        print('{} images found.'.format(self.length))

    def __len__(self):
        return self.length

    def set_epoch(self,epoch):
        random.seed(1234+epoch)

    def __getitem__(self, index):
        index=self.random[index]
        img_path = os.path.join(self.data_dir, self.img_list[index])

        id=self.img_list[index].split('/')[-1]

        
            
        if self.vis:
            rgb = cv2.imread(img_path + '_color.png')[:, :, :3]
            image=rgb.copy()

        # rgb = rgb[:, :, ::-1]

        depth = load_depth(img_path)

        mask = cv2.imread(img_path + '_mask.png')[:, :, 2]

        coord = cv2.imread(img_path + '_coord.png')[:, :, :3]
        coord = coord[:, :, (2, 1, 0)]
        coord = np.array(coord, dtype=np.float32) / 255
        coord[:, :, 2] = 1 - coord[:, :, 2]

        with open(img_path + '_label.pkl', 'rb') as f:
            gts = cPickle.load(f)
        if 'CAMERA' in img_path.split('/'):
            cam_fx, cam_fy, cam_cx, cam_cy = self.camera_intrinsics
        else:
            cam_fx, cam_fy, cam_cx, cam_cy = self.real_intrinsics

        cam_K=np.identity(3, dtype=np.float32)
        cam_K[0,0],cam_K[1,1],cam_K[0,2],cam_K[1,2]=cam_fx, cam_fy, cam_cx, cam_cy

        # select one foreground object
        ''''''
        idx = random.randint(0, len(gts['instance_ids'])-1)
        if self.per_obj is not None:
            for i in range(len(gts['instance_ids'])):
                if gts['class_ids'][i]-1==self.per_obj:
                    idx=i

        cat_id=gts['class_ids'][idx]-1
        inst_id = gts['instance_ids'][idx]
        rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][idx])
        # sample points from mask
        mask = np.equal(mask, inst_id)
        mask = np.logical_and(mask, depth > 0)
        mask = mask.flatten()

        depth_masked=(depth.flatten())[mask]  #N
        xmap_masked=(self.xmap.flatten())[mask]
        ymap_masked=(self.ymap.flatten())[mask]
        
        pt2=depth_masked/self.norm_scale
        pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
        pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
        points=np.stack((pt0,pt1,pt2),axis=1)  #N,3

        l_all=points.shape[0]

        if l_all>=self.n_pts:
            choose=np.random.choice(l_all,self.n_pts,replace=False)
        else:
            choose=np.random.choice(l_all,self.n_pts,replace=True)

        nocs = coord.reshape(-1,3)[mask,...][choose, :] - 0.5
        
        points=points[choose,...]



        crop_w = rmax - rmin
        ratio = self.img_size / crop_w
        col_idx = choose % crop_w
        row_idx = choose // crop_w
        choose = (np.floor(row_idx * ratio) * self.img_size + np.floor(col_idx * ratio)).astype(np.int64)


        scale = gts['scales'][idx]
        rotation = gts['rotations'][idx]
        translation = gts['translations'][idx]
        prior = self.mean_shapes[cat_id].astype(np.float32)
  
        # adjust nocs coords for mug category
        if cat_id==5:
            T0 = self.mug_meta[gts['model_list'][idx]][0]
            s0 = self.mug_meta[gts['model_list'][idx]][1]
            nocs = s0 * (nocs + T0)
        
        # map ambiguous rotation to canonical rotation
        if cat_id in self.sym_ids:
            rotation = gts['rotations'][idx]
            # assume continuous axis rotation symmetry
            theta_x = rotation[0, 0] + rotation[2, 2]
            theta_y = rotation[0, 2] - rotation[2, 0]
            r_norm = math.sqrt(theta_x**2 + theta_y**2)
            s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                            [0.0,            1.0,  0.0           ],
                            [theta_y/r_norm, 0.0,  theta_x/r_norm]])
            rotation = rotation @ s_map
            nocs = nocs @ s_map
        
        sRT = np.identity(4, dtype=np.float32)
        RT=np.identity(4,dtype=np.float32)
        sRT[:3, :3] = scale * rotation
        sRT[:3, 3] = translation
        RT[:3,:3]=rotation
        RT[:3,3]=translation

        model = self.models[gts['model_list'][idx]].astype(np.float32)


        model=torch.as_tensor(model.astype(np.float32))
        points=torch.as_tensor(points.astype(np.float32))
        R=torch.as_tensor(rotation.astype(np.float32))
        t=torch.as_tensor(translation.astype(np.float32))
        s=torch.as_tensor(scale.astype(np.float32))
        nocs=torch.as_tensor(nocs.astype(np.float32))

        sym_info = self.get_sym_info(cat_id, mug_handle=1)
        bb_aug, rt_aug_t, rt_aug_R = self.generate_aug_parameters()
        dimension_delta,mean_shape=self.get_fs_net_scale( model, s,cat_id)

        sym_info=torch.as_tensor(sym_info.astype(np.float32)).contiguous()
        bb_aug, rt_aug_t, rt_aug_R=torch.as_tensor(bb_aug, dtype=torch.float32).contiguous(),torch.as_tensor(rt_aug_t, dtype=torch.float32).contiguous(),torch.as_tensor(rt_aug_R, dtype=torch.float32).contiguous()
        dimension_delta=torch.as_tensor(dimension_delta,dtype=torch.float32).contiguous()
        mean_shape=torch.as_tensor(mean_shape,dtype=torch.float32).contiguous()

        if self.use_augment:
            points, R, t, dimension, model, nocs,s=self.data_augment(points,R,t,dimension_delta+mean_shape,sym_info,bb_aug,rt_aug_t,rt_aug_R,model,s,nocs,cat_id)

            dimension_delta=dimension-mean_shape

            if cat_id in self.sym_ids:
                # assume continuous axis rotation symmetry
                R=R.numpy()
                nocs=nocs.numpy()
                theta_x = R[0, 0] + R[2, 2]
                theta_y = R[0, 2] - R[2, 0]
                r_norm = math.sqrt(theta_x**2 + theta_y**2)
                s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                                [0.0,            1.0,  0.0           ],
                                [theta_y/r_norm, 0.0,  theta_x/r_norm]])
                R = R @ s_map
                nocs = nocs @ s_map
                R=torch.as_tensor(R.astype(np.float32))
                nocs=torch.as_tensor(nocs.astype(np.float32))

        gt_green,gt_red=self.get_gt_v(R)
        

        #data=data[choose,...]
        data_dict={}
        if self.mode=='test':
            data_dict['handle_visiblity']=gts['handle_visibility'][idx]
        data_dict['points']=points.contiguous()
        data_dict['nocs']=nocs.contiguous()
        data_dict['prior']=torch.as_tensor(prior).contiguous()
        data_dict['cat_id']=torch.as_tensor(cat_id)
        data_dict['R']=R.contiguous()
        data_dict['t']=t.contiguous()
        data_dict['s']=s.contiguous()
        data_dict['gt_green']=gt_green.contiguous()
        data_dict['gt_red']=gt_red.contiguous()
        data_dict['dimension_delta']=dimension_delta.contiguous()
        data_dict['mean_shape']=mean_shape.contiguous()
        data_dict['sym']=sym_info.contiguous()
        if self.vis:
            data_dict['RT']=torch.as_tensor(RT.astype(np.float32)).contiguous()
            data_dict['id']=id
            data_dict['sRT']=torch.as_tensor(sRT.astype(np.float32)).contiguous()
            data_dict['image']=image
            data_dict['cam_K']=cam_K
        data_dict['model']=model.contiguous()
        data_dict['img_path']=img_path

        return data_dict

    @torch.no_grad()
    def get_gt_v(self,Rs, axis=2):
        # TODO use 3 axis, the order remains: do we need to change order?
        if axis == 3:
            raise NotImplementedError
        else:
            assert axis == 2
            gt_green = Rs[:,1:2]
            gt_red = Rs[:,0:1]
        return gt_green, gt_red


    @torch.no_grad()
    def data_augment(self,PC,gt_R,gt_t,gt_s,sym,aug_bb,aug_rt_t,aug_rt_r,model_point,nocs_scale,PC_nocs,obj_id):
        prop_bb = torch.rand(1)
        if prop_bb < 0.3:
            #   R, t, s, s_x=(0.9, 1.1), s_y=(0.9, 1.1), s_z=(0.9, 1.1), sym=None
            PC_new, gt_s_new, nocs_new, model_new,nocs_scale_aug = defor_3D_bb(PC, gt_R,
                                            gt_t, gt_s, PC_nocs, model_point,
                                            sym=sym, aug_bb=aug_bb)
            PC = PC_new
            gt_s = gt_s_new
            PC_nocs = nocs_new
            model_point = model_new
            nocs_scale=nocs_scale/nocs_scale_aug


        prop_rt = torch.rand(1)
        if prop_rt < 0.3:
            PC_new, gt_R_new, gt_t_new = defor_3D_rt(PC, gt_R,
                                                        gt_t, aug_rt_t, aug_rt_r)
            PC = PC_new
            gt_R = gt_R_new
            gt_t = gt_t_new.view(-1)

        prop_bc = torch.rand(1)
        # only do bc for mug and bowl
        b=False
        if prop_bc < 0.3 and (obj_id in [1,5]):
            b=True
            PC_new, gt_s_new, model_point_new, nocs_new,nocs_scale_aug = defor_3D_bc(PC, gt_R, gt_t,gt_s,model_point, nocs_scale, PC_nocs)
            PC = PC_new
            gt_s = gt_s_new
            model_point = model_point_new
            PC_nocs = nocs_new
            nocs_scale=nocs_scale/nocs_scale_aug

        prop_nl = torch.rand(1)
        if not b and prop_nl < 0.3 and (obj_id in [0,1,2,3,5]):
            if obj_id in [0,1,3,5]:
                sel_axis = 1
            elif obj_id in [2]:
                sel_axis = 0
            else:
                sel_axis = None

            PC_new, gt_s_new, model_point_new, nocs_new,nocs_scale_aug = deform_non_linear(PC, gt_R, gt_t,gt_s,PC_nocs, model_point, sel_axis)

            PC = PC_new
            gt_s = gt_s_new
            model_point = model_point_new
            PC_nocs = nocs_new
            nocs_scale=nocs_scale/nocs_scale_aug


        prop_pc = torch.rand(1)
        if prop_pc < 0.3:
            PC_new = defor_3D_pc(PC, 0.001)
            PC = PC_new

        pro_aug=torch.rand(1)
        if pro_aug<0.1:
            num=random.randint(1,10)
            position=list(range(1024))
            position=random.sample(position,num)
            position=torch.tensor(position,dtype=torch.long)
            PC[position,...]=torch.rand((num,3))*gt_s*0.6+gt_t
        
        #  augmentation finish
        return PC, gt_R, gt_t, gt_s, model_point, PC_nocs,nocs_scale


    def get_sym_info(self, c, mug_handle=1):
        #  sym_info  c0 : face classfication  c1, c2, c3:Three view symmetry, correspond to xy, xz, yz respectively
        # c0: 0 no symmetry 1 axis symmetry 2 two reflection planes 3 unimplemented type
        #  Y axis points upwards, x axis pass through the handle, z axis otherwise
        #
        # for specific defination, see sketch_loss
        if c == 0:#'bottle'
            sym = np.array([1, 1, 0, 1], dtype=np.int32)
        elif c == 1:#'bowl'
            sym = np.array([1, 1, 0, 1], dtype=np.int32)
        elif c == 2:#'camera'
            sym = np.array([0, 0, 0, 0], dtype=np.int32)
        elif c == 3:#'can'
            sym = np.array([1, 1, 1, 1], dtype=np.int32)
        elif c == 4:#'laptop'
            sym = np.array([0, 1, 0, 0], dtype=np.int32)
        elif c ==  5 and mug_handle == 1:#'mug'
            sym = np.array([0, 1, 0, 0], dtype=np.int32)  # for mug, we currently mark it as no symmetry
        elif c == 5 and mug_handle == 0:#'mug'
            sym = np.array([1, 0, 0, 0], dtype=np.int32)
        else:
            sym = np.array([0, 0, 0, 0], dtype=np.int32)
        return sym


    def generate_aug_parameters(self, s_x=(0.8, 1.2), s_y=(0.8, 1.2), s_z=(0.8, 1.2), ax=50, ay=50, az=50, a=15):
        # for bb aug
        ex, ey, ez = np.random.rand(3)
        ex = ex * (s_x[1] - s_x[0]) + s_x[0]
        ey = ey * (s_y[1] - s_y[0]) + s_y[0]
        ez = ez * (s_z[1] - s_z[0]) + s_z[0]
        # for R, t aug
        Rm = get_rotation(np.random.uniform(-a, a), np.random.uniform(-a, a), np.random.uniform(-a, a))
        dx = np.random.rand() * 2 * ax - ax
        dy = np.random.rand() * 2 * ay - ay
        dz = np.random.rand() * 2 * az - az
        return np.array([ex, ey, ez], dtype=np.float32), np.array([dx, dy, dz], dtype=np.float32) / 1000.0, Rm

    
    def get_fs_net_scale(self, model, nocs_scale,c):
        # model pc x 3
        lx = 2 * max(max(model[:, 0]), -min(model[:, 0]))
        ly = max(model[:, 1]) - min(model[:, 1])
        lz = max(model[:, 2]) - min(model[:, 2])

        # real scale
        lx_t = lx * nocs_scale * 1000
        ly_t = ly * nocs_scale * 1000
        lz_t = lz * nocs_scale * 1000

        if c == 0:#'bottle'
            unitx = 87
            unity = 220
            unitz = 89
        elif c == 1:#'bowl'
            unitx = 165
            unity = 80
            unitz = 165
        elif c == 2:#'camera'
            unitx = 88
            unity = 128
            unitz = 156
        elif c == 3:#'can'
            unitx = 68
            unity = 146
            unitz = 72
        elif c == 4:#'laptop'
            unitx = 346
            unity = 200
            unitz = 335
        elif c == 5:#'mug'
            unitx = 146
            unity = 83
            unitz = 114
        elif c == '02876657':
            unitx = 324 / 4
            unity = 874 / 4
            unitz = 321 / 4
        elif c == '02880940':
            unitx = 675 / 4
            unity = 271 / 4
            unitz = 675 / 4
        elif c == '02942699':
            unitx = 464 / 4
            unity = 487 / 4
            unitz = 702 / 4
        elif c == '02946921':
            unitx = 450 / 4
            unity = 753 / 4
            unitz = 460 / 4
        elif c == '03642806':
            unitx = 581 / 4
            unity = 445 / 4
            unitz = 672 / 4
        elif c == '03797390':
            unitx = 670 / 4
            unity = 540 / 4
            unitz = 497 / 4
        else:
            unitx = 0
            unity = 0
            unitz = 0
            print('This category is not recorded in my little brain.')
            raise NotImplementedError
        # scale residual
        return np.array([lx_t - unitx, ly_t - unity, lz_t - unitz])/1000.0, np.array([unitx, unity, unitz])/1000.0

