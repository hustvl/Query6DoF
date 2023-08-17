import sys
sys.path.append('.')
import os
import time
import argparse
import cv2
import math
import glob
import numpy as np
from tqdm import tqdm
import _pickle as cPickle
import torch
import torch.nn.functional as F
from mmengine import Config,DictAction
from network import NETWORK_REGISTRY
from utils import load_depth, get_bbox, compute_mAP, plot_mAP
from utils.logging import create_checkpoint
import random
from utils.utils import farthest_point_sample,index_points

def set_random_seed(seed, deterministic=False): 
    random.seed(seed) 
    np.random.seed(seed) 
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    if deterministic: 
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='real_test', help='val, real_test')
parser.add_argument('--data_dir', type=str, default='data', help='data directory')
parser.add_argument('--n_cat', type=int, default=6, help='number of object categories')
parser.add_argument('--nv_prior', type=int, default=1024, help='number of vertices in shape priors')
parser.add_argument('--model', type=str, default='results/camera/model_50.pth', help='resume from saved model')
parser.add_argument('--n_pts', type=int, default=1024, help='number of foreground points')
parser.add_argument('--img_size', type=int, default=192, help='cropped image size')
parser.add_argument('--gpus', type=str, default='1', help='GPU to use')
parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)
parser.add_argument('--cfg-options',
                    nargs='+',
                    action=DictAction,
                    help='override some settings in the used config, the key-value pair '
                    'in xxx=yyy format will be merged into config file. If the value to '
                    'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
                    'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
                    'Note that the quotation marks are necessary and that no white space '
                    'is allowed.')
opt = parser.parse_args()

per_obj=None
use_gt_mask=False
mean_shapes = np.load('assets/mean_points_emb.npy')


result_dir='results/eval_real'


xmap = np.array([[i for i in range(640)] for j in range(480)])
ymap = np.array([[j for i in range(640)] for j in range(480)])
norm_scale = 1000.0

@torch.inference_mode()
def detect():
    # resume model
    print('use_gt_mask: ',use_gt_mask)
    global opt
    global result_dir
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
    opt = Config.fromfile(opt.cfg)
    assert not opt.train
    assert opt.DATA in ['val', 'real_test']
    opt.MODEL.decoder.training=False
    opt.MODEL.training=False
    if opt.DATA == 'val':
        result_dir = 'results/eval_camera'
        file_path = 'CAMERA/val_list.txt'
        cam_fx, cam_fy, cam_cx, cam_cy = 577.5, 577.5, 319.5, 239.5
    else:
        result_dir = 'results/eval_real'
        file_path = 'Real/test_list.txt'
        cam_fx, cam_fy, cam_cx, cam_cy = 591.0125, 590.16775, 322.525, 244.11084
        cam_K=np.identity(3, dtype=np.float32)
        cam_K[0,0],cam_K[1,1],cam_K[0,2],cam_K[1,2]=cam_fx, cam_fy, cam_cx, cam_cy

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    set_random_seed(123)

    mean_shapes = np.load('assets/mean_points_emb.npy')

    model = NETWORK_REGISTRY.build(opt.MODEL)
    #model.init_para(0)
    model = torch.nn.DataParallel(model).cuda()
    final_output_dir = create_checkpoint(opt,None)
    checkpoint_file = os.path.join(
            final_output_dir, 'model', opt.RESUME_FILE)
    
    checkpoint = torch.load(checkpoint_file, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()


    # get test data list
    img_list = [os.path.join(file_path.split('/')[0], line.rstrip('\n'))
                for line in open(os.path.join(opt.DATASET.data_dir, file_path))]
    # frame by frame test
    times=[]
    t_inference = 0.0
    t_umeyama = 0.0
    inst_count = 0
    img_count = 0
    t_start = time.time()

    # img_list=img_list[:300]
    cam_K=np.identity(3, dtype=np.float32)
    cam_K[0,0],cam_K[1,1],cam_K[0,2],cam_K[1,2]=cam_fx, cam_fy, cam_cx, cam_cy
    for path in tqdm(img_list):
        img_path = os.path.join(opt.DATASET.data_dir, path)
        raw_depth = load_depth(img_path)
        # load mask-rcnn detection results
        img_path_parsing = img_path.split('/')
        
        if use_gt_mask:
            gt_mask=cv2.imread(img_path + '_mask.png')[:, :, 2]
            with open(img_path + '_label.pkl', 'rb') as f:
                gts = cPickle.load(f)
            num_insts=len(gts['instance_ids'])
        else:
            
            mrcnn_path = os.path.join('data/results/mrcnn_results', opt.DATA, 'results_{}_{}_{}.pkl'.format(
            opt.DATA.split('_')[-1], img_path_parsing[-2], img_path_parsing[-1]))
            with open(mrcnn_path, 'rb') as f:
                mrcnn_result = cPickle.load(f)
            num_insts = len(mrcnn_result['class_ids'])


        f_sRT = np.zeros((num_insts, 4, 4), dtype=float)
        f_size = np.zeros((num_insts, 3), dtype=float)
  
        # prepare frame data
        f_points, f_catId,f_rgb,f_mask,f_choose,f_prior,f_sym,f_mean_shape= [], [],[],[],[],[],[],[]
        valid_inst = []
        
        for i in range(num_insts):
            if use_gt_mask:
                new_gt_mask=gt_mask.copy()
                new_gt_mask=np.equal(new_gt_mask,gts['instance_ids'][i])
                new_gt_mask=np.logical_and(new_gt_mask,raw_depth>0)
                mask=new_gt_mask
                cat_id=gts['class_ids'][i]-1
            else:
                cat_id = mrcnn_result['class_ids'][i] - 1
                mask = np.logical_and(mrcnn_result['masks'][:, :, i], raw_depth > 0)
            if per_obj is not None and  cat_id not in per_obj:
                continue   
            # raw_rgb[mask,:]=255

            # if cat_id==1:
            #     raw_rgb[mask,:]=255
            #     cv2.imwrite(os.path.join('imgs','pred_masks',img_path_parsing[-1]+'_1.png'),raw_rgb)

            mask = mask.flatten()

            depth_masked=(raw_depth.flatten())[mask]  #N
            xmap_masked=(xmap.flatten())[mask]
            ymap_masked=(ymap.flatten())[mask]

            pt2 = depth_masked / norm_scale
            pt0 = (xmap_masked - cam_cx) * pt2 / cam_fx
            pt1 = (ymap_masked - cam_cy) * pt2 / cam_fy
            points = np.stack((pt0, pt1, pt2), axis=1)
            
            l_all=points.shape[0]

            if l_all < 32:
                f_sRT[i] = np.identity(4, dtype=float)
                prior = mean_shapes[cat_id].astype(np.float32)
                f_size[i] = 2 * np.amax(np.abs(prior), axis=0)
                continue
            else:
                valid_inst.append(i)
            #prior = mean_shapes[cat_id].astype(np.float32)
            if use_gt_mask:
                rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][i])
            else:
                rmin, rmax, cmin, cmax = get_bbox(mrcnn_result['rois'][i])


            # process objects with valid depth observation
            if l_all >= opt.DATASET.n_pts:
                choose=np.random.choice(l_all,opt.DATASET.n_pts,replace=False)
            else:
                choose=np.random.choice(l_all,opt.DATASET.n_pts,replace=True)

            points=points[choose,...]


            sym_info = get_sym_info(cat_id, mug_handle=1)
            mean_shape=get_fs_net_scale(cat_id)

            # concatenate instances
            f_points.append(points)
            f_catId.append(cat_id)
            # f_rgb.append(rgb)
            f_mask.append(mask)
            # f_choose.append(choose)
            f_sym.append(sym_info)
            f_mean_shape.append(mean_shape)
            #f_prior.append(prior)
        if len(valid_inst):
            f_points = torch.cuda.FloatTensor(np.array(f_points)).contiguous()
            f_catId = torch.cuda.LongTensor(np.array(f_catId)).contiguous()
            # f_rgb=torch.cuda.FloatTensor(np.array(f_rgb)).contiguous()
            f_choose=torch.cuda.LongTensor(np.array(f_choose)).contiguous()
            f_sym=torch.cuda.LongTensor(np.array(f_sym)).contiguous()
            f_mean_shape=torch.cuda.FloatTensor(np.array(f_mean_shape)).contiguous()
            batched_input={
                'points':f_points,
                'cat_id':f_catId,
                'sym':f_sym,
                'mean_shape':f_mean_shape
                #'prior':f_prior
            }
            # inference
            torch.cuda.synchronize()
            t_now = time.time()
            pred_sRT, size = model(batched_input)
            for i in range(len(valid_inst)):
                inst_idx = valid_inst[i]
                f_sRT[inst_idx] = pred_sRT[i]
                f_size[inst_idx]=size[i]
            torch.cuda.synchronize()
            inference=time.time() - t_now
            times.append(inference)
            t_inference += (inference)
            img_count += 1
            inst_count += len(valid_inst)


        # save results
        result = {}
        if not use_gt_mask:
            with open(img_path + '_label.pkl', 'rb') as f:
                gts = cPickle.load(f)
        result['gt_class_ids'] = gts['class_ids']
        result['gt_bboxes'] = gts['bboxes']
        for idx,cat_id in enumerate(gts['class_ids']):
            cat_id=cat_id-1
            assert cat_id>=0
            rotation = gts['rotations'][idx]
            scale = gts['scales'][idx]
            translation = gts['translations'][idx]
            if cat_id in [0, 1, 3]:
                # assume continuous axis rotation symmetry
                theta_x = rotation[0, 0] + rotation[2, 2]
                theta_y = rotation[0, 2] - rotation[2, 0]
                r_norm = math.sqrt(theta_x**2 + theta_y**2)
                s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                                [0.0,            1.0,  0.0           ],
                                [theta_y/r_norm, 0.0,  theta_x/r_norm]])
                rotation = rotation @ s_map
                sRT = np.identity(4, dtype=np.float32)
                sRT[:3, :3] = scale * rotation
                sRT[:3, 3] = translation
                gts['poses'][idx]=sRT
        result['gt_RTs'] = gts['poses']
        result['gt_scales'] = gts['size']
        result['gt_handle_visibility'] = gts['handle_visibility']

        if use_gt_mask:
            result['pred_class_ids'] = gts['class_ids']
            result['pred_bboxes'] = gts['bboxes']
            result['pred_scores'] = np.ones((num_insts,))
        else:
            result['pred_class_ids'] = mrcnn_result['class_ids']
            result['pred_bboxes'] = mrcnn_result['rois']
            result['pred_scores'] = mrcnn_result['scores']

        
        result['pred_RTs'] = f_sRT
        result['pred_scales']=f_size

        image_short_path = '_'.join(img_path_parsing[-3:])
        save_path = os.path.join(result_dir, 'results_{}.pkl'.format(image_short_path))
        with open(save_path, 'wb') as f:
            cPickle.dump(result, f)


        
    # write statistics    
    total_time=0.0
    times=times[100:]
    for t in times:
        total_time+=t
    fw = open('{0}/eval_logs.txt'.format(result_dir), 'w')
    messages = []
    messages.append("Total images: {}".format(len(img_list)))
    messages.append("Valid images: {},  Total instances: {},  Average: {:.2f}/image".format(
        img_count, inst_count, inst_count/img_count))
    messages.append("Inference time: {:06f}  Average: {:06f}/image  fps:{:06f}".format(t_inference, total_time/(img_count-100),(img_count-100)/total_time))
    messages.append("Total time: {:06f}".format(time.time() - t_start))
    for msg in messages:
        print(msg)
        fw.write(msg + '\n')
    fw.close()
    del model


def evaluate():
    degree_thres_list = list(range(0, 61, 1))
    shift_thres_list = [i / 2 for i in range(21)]
    iou_thres_list = [i / 100 for i in range(101)]
    # predictions
    result_pkl_list = glob.glob(os.path.join(result_dir, 'results_*.pkl'))
    result_pkl_list = sorted(result_pkl_list)

    # result_pkl_list=result_pkl_list[:100]
    assert len(result_pkl_list)
    pred_results = []
    for pkl_path in result_pkl_list:
        with open(pkl_path, 'rb') as f:
            result = cPickle.load(f)
            if 'gt_handle_visibility' not in result:
                result['gt_handle_visibility'] = np.ones_like(result['gt_class_ids'])
            else:
                assert len(result['gt_handle_visibility']) == len(result['gt_class_ids']), "{} {}".format(
                    result['gt_handle_visibility'], result['gt_class_ids'])
        if type(result) is list:
            pred_results += result
        elif type(result) is dict:
            pred_results.append(result)
        else:
            assert False

    # To be consistent with NOCS, set use_matches_for_pose=True for mAP evaluation
    iou_aps, pose_aps, iou_acc, pose_acc = compute_mAP(pred_results, result_dir, degree_thres_list, shift_thres_list,
                                                       iou_thres_list, iou_pose_thres=0.1, use_matches_for_pose=True)
    #print(pose_aps)
    # np.save('pose_aps', pose_aps, allow_pickle=True, fix_imports=True)
    
    # metric
    fw = open('{0}/eval_logs.txt'.format(result_dir), 'a')
    iou_25_idx = iou_thres_list.index(0.25)
    iou_50_idx = iou_thres_list.index(0.5)
    iou_75_idx = iou_thres_list.index(0.75)
    degree_05_idx = degree_thres_list.index(5)
    degree_10_idx = degree_thres_list.index(10)
    shift_02_idx = shift_thres_list.index(2)
    shift_05_idx = shift_thres_list.index(5)
    messages = []
    messages.append('mAP:')
    messages.append('3D IoU at 25: {:.1f}'.format(iou_aps[-1, iou_25_idx] * 100))
    messages.append('3D IoU at 50: {:.1f}'.format(iou_aps[-1, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.1f}'.format(iou_aps[-1, iou_75_idx] * 100))
    messages.append('5 degree, 2cm: {:.1f}'.format(pose_aps[-1, degree_05_idx, shift_02_idx] * 100))
    messages.append('5 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_05_idx, shift_05_idx] * 100))
    messages.append('10 degree, 2cm: {:.1f}'.format(pose_aps[-1, degree_10_idx, shift_02_idx] * 100))
    messages.append('10 degree, 5cm: {:.1f}'.format(pose_aps[-1, degree_10_idx, shift_05_idx] * 100))
    messages.append('Acc:')
    messages.append('3D IoU at 25: {:.1f}'.format(iou_acc[-1, iou_25_idx] * 100))
    messages.append('3D IoU at 50: {:.1f}'.format(iou_acc[-1, iou_50_idx] * 100))
    messages.append('3D IoU at 75: {:.1f}'.format(iou_acc[-1, iou_75_idx] * 100))
    messages.append('5 degree, 2cm: {:.1f}'.format(pose_acc[-1, degree_05_idx, shift_02_idx] * 100))
    messages.append('5 degree, 5cm: {:.1f}'.format(pose_acc[-1, degree_05_idx, shift_05_idx] * 100))
    messages.append('10 degree, 2cm: {:.1f}'.format(pose_acc[-1, degree_10_idx, shift_02_idx] * 100))
    messages.append('10 degree, 5cm: {:.1f}'.format(pose_acc[-1, degree_10_idx, shift_05_idx] * 100))
    for msg in messages:
        print(msg)
        fw.write(msg + '\n')
    fw.close()
    # load NOCS results
    pkl_path = os.path.join('results/nocs_results', opt.DATA, 'mAP_Acc.pkl')
    with open(pkl_path, 'rb') as f:
        nocs_results = cPickle.load(f)
    nocs_iou_aps = nocs_results['iou_aps'][-1, :]
    nocs_pose_aps = nocs_results['pose_aps'][-1, :, :]
    iou_aps = np.concatenate((iou_aps, nocs_iou_aps[None, :]), axis=0)
    pose_aps = np.concatenate((pose_aps, nocs_pose_aps[None, :, :]), axis=0)
    # plot
    plot_mAP(iou_aps, pose_aps, result_dir, iou_thres_list, degree_thres_list, shift_thres_list)


def get_sym_info(c, mug_handle=1):
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


def get_fs_net_scale(c):
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
        return  np.array([unitx, unity, unitz])/1000.0


if __name__ == '__main__':
    print('Detecting ...')
    detect()
    print('Evaluating ...')
    evaluate()