import numpy as np
import pickle

from tqdm  import tqdm
from features.kinetic import extract_kinetic_features
from features.manual_new import extract_manual_features
from scipy import linalg
# kinetic, manual
import torch
import os, sys
import argparse
# from render import ax_to_6v
sys.path.append("/data2/lrh/project/dance/LodgePlus/Lodge_plus_exp/DanceDiffusion")
from dld.data.utils.smplfk import SMPLX_Skeleton
from dld.data.utils.smplfk import do_smplxfk
from dld.data.utils.motion_process import recover_from_ric266v_grad, recover_from_ricv_norotinit_grad, recover_from_ric266

def normalize(feat, feat2):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    
    return (feat - mean) / (std + 1e-10), (feat2 - mean) / (std + 1e-10)


def normalize_one(feat):
    mean = feat.mean(axis=0)
    std = feat.std(axis=0)
    
    return (feat - mean) / (std + 1e-10)

def quantized_metrics(predicted_pkl_root, gt_pkl_root):
    pred_features_k = []
    pred_features_m = []
    gt_freatures_k = []
    gt_freatures_m = []

    # for pkl in os.listdir(predicted_pkl_root):
    #     pred_features_k.append(np.load(os.path.join(predicted_pkl_root, 'kinetic_features', pkl))) 
    #     pred_features_m.append(np.load(os.path.join(predicted_pkl_root, 'manual_features_new', pkl)))
    #     gt_freatures_k.append(np.load(os.path.join(predicted_pkl_root, 'kinetic_features', pkl)))
    #     gt_freatures_m.append(np.load(os.path.join(predicted_pkl_root, 'manual_features_new', pkl)))

    pred_features_k = [np.load(os.path.join(predicted_pkl_root, 'kinetic_features', pkl)) for pkl in os.listdir(os.path.join(predicted_pkl_root, 'kinetic_features'))]
    pred_features_m = [np.load(os.path.join(predicted_pkl_root, 'manual_features_new', pkl)) for pkl in os.listdir(os.path.join(predicted_pkl_root, 'manual_features_new'))]
    
    gt_freatures_k = [np.load(os.path.join(gt_pkl_root, 'kinetic_features', pkl)) for pkl in os.listdir(os.path.join(gt_pkl_root, 'kinetic_features'))]
    gt_freatures_m = [np.load(os.path.join(gt_pkl_root, 'manual_features_new', pkl)) for pkl in os.listdir(os.path.join(gt_pkl_root, 'manual_features_new'))]
    
    
    pred_features_k = np.stack(pred_features_k)  # Nx72 p40
    pred_features_m = np.stack(pred_features_m) # Nx32
    gt_freatures_k = np.stack(gt_freatures_k) # N' x 72 N' >> N
    gt_freatures_m = np.stack(gt_freatures_m) # 
    if gt_freatures_k.shape[1] == 72:
        gt_freatures_k = gt_freatures_k[:,:66]
    if pred_features_k.shape[1] == 72:
        pred_features_k = pred_features_k[:,:66]

#   T x 24 x 3 --> 72
# T x72 -->32 
    # print(gt_freatures_k.mean(axis=0))
    # print(pred_features_k.mean(axis=0))
    # print(gt_freatures_m.mean(axis=0))
    # print(pred_features_m.mean(axis=0))
    # print(gt_freatures_k.std(axis=0))
    # print(pred_features_k.std(axis=0))
    # print(gt_freatures_m.std(axis=0))
    # print(pred_features_m.std(axis=0))

    # gt_freatures_k = normalize_one(gt_freatures_k)
    # gt_freatures_m = normalize_one(gt_freatures_m) 
    # pred_features_k = normalize_one(pred_features_k)
    # pred_features_m = normalize_one(pred_features_m)     
    
    gt_freatures_k, pred_features_k = normalize(gt_freatures_k, pred_features_k)
    gt_freatures_m, pred_features_m = normalize(gt_freatures_m, pred_features_m) 
    # # pred_features_k = normalize(pred_features_k)
    # pred_features_m = normalize(pred_features_m) 
    # pred_features_k = normalize(pred_features_k)
    # pred_features_m = normalize(pred_features_m)
    
    # print(gt_freatures_k.mean(axis=0))
    print(pred_features_k.mean(axis=0))
    # print(gt_freatures_m.mean(axis=0))
    print(pred_features_m.mean(axis=0))
    # print(gt_freatures_k.std(axis=0))
    print(pred_features_k.std(axis=0))
    # print(gt_freatures_m.std(axis=0))
    print(pred_features_m.std(axis=0))

    
    # print(gt_freatures_k)
    # print(gt_freatures_m)

    print('Calculating metrics')

    fid_k = calc_fid(pred_features_k, gt_freatures_k)
    fid_m = calc_fid(pred_features_m, gt_freatures_m)
    # div_k_gt = '***'
    # div_m_gt = '***'
    div_k_gt = calculate_avg_distance(gt_freatures_k)
    div_m_gt = calculate_avg_distance(gt_freatures_m)
    div_k = calculate_avg_distance(pred_features_k)
    div_m = calculate_avg_distance(pred_features_m)


    metrics = {'fid_k': fid_k, 'fid_m': fid_m, 'div_k': div_k, 'div_m' : div_m, 'div_k_gt': div_k_gt, 'div_m_gt': div_m_gt}
    return metrics


def calc_fid(kps_gen, kps_gt):

    print(kps_gen.shape)
    print(kps_gt.shape)

    # kps_gen = kps_gen[:20, :]

    mu_gen = np.mean(kps_gen, axis=0)
    sigma_gen = np.cov(kps_gen, rowvar=False)

    mu_gt = np.mean(kps_gt, axis=0)
    sigma_gt = np.cov(kps_gt, rowvar=False)

    mu1,mu2,sigma1,sigma2 = mu_gen, mu_gt, sigma_gen, sigma_gt

    diff = mu1 - mu2
    eps = 1e-5
    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            # raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calc_diversity(feats):
    feat_array = np.array(feats)
    n, c = feat_array.shape
    diff = np.array([feat_array] * n) - feat_array.reshape(n, 1, c)
    return np.sqrt(np.sum(diff**2, axis=2)).sum() / n / (n-1)

def calculate_avg_distance(feature_list, mean=None, std=None):
    feature_list = np.stack(feature_list)
    n = feature_list.shape[0]
    # normalize the scale
    if (mean is not None) and (std is not None):
        feature_list = (feature_list - mean) / std
    dist = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist += np.linalg.norm(feature_list[i] - feature_list[j])
    dist /= (n * n - n) / 2
    return dist

def calc_and_save_feats(root):
    if not os.path.exists(os.path.join(root, 'kinetic_features')):
        os.mkdir(os.path.join(root, 'kinetic_features'))
    if not os.path.exists(os.path.join(root, 'manual_features_new')):
        os.mkdir(os.path.join(root, 'manual_features_new'))
    
    # gt_list = []
    pred_list = []
    test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211", "193", "179", "065", "137", "161", "092", "120", "037", "109", "204", "144"]
    # test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211", "193", "179", "065", "137", "161", "092",  "037", "109", "204", "144"]
    for pkl in tqdm(os.listdir(root)):
        if os.path.isdir(os.path.join(root, pkl)):
            continue
        if pkl[0] == 'M':
            continue
        # if pkl[:3] == 'gpt':
        #     continue
        if pkl.split('.')[-1] not in ['pkl','npy']:
            continue
        # if pkl[:3] == 'dod':
        #     continue
        # if pkl[:3] not in test_list:
        #     continue
        if pkl[-3:] == 'pkl':
            data = pickle.load(open(os.path.join(root, pkl), "rb"))
            print(data.keys())
            model_q = torch.from_numpy(data['smpl_poses'] ).to(device)   
            model_x = torch.from_numpy(data['smpl_trans'] ).to(device)    
            print("model_q", model_q.shape)
            print("model_x", model_x.shape)
            model_q156 = torch.cat([model_q, torch.zeros([model_q.shape[0], 90]).to(device) ], dim=-1)
            with torch.no_grad():
                joint3d = smplx_model.forward(model_q156, model_x)[:,:24,:]
                print("joint3d", joint3d.shape)
        elif pkl[-3:] == 'npy':
            if pkl[0] == 'M':
                continue
            
            data = np.load(os.path.join(root, pkl))
            if len(data.shape) == 3:
                print('attention!!, len(data.shape) = 3')
                data = data.reshape(-1, data.shape[2])
            if data.shape[1] == 139 or data.shape[1] == 319:
                data = data[:1024,:139]
                data = torch.from_numpy(data).to(device)   
                with torch.no_grad():
                    joint3d = do_smplxfk(data, smplx_model)[:,:24,:]
            elif data.shape[1] == 135 or data.shape[1] == 315:
                data = data[:1024,:135]
                data = torch.from_numpy(data).to(device)   
                data = torch.cat([torch.zeros([data.shape[0], 4]).to(data),  data], dim=1) 
                with torch.no_grad():
                    joint3d = do_smplxfk(data, smplx_model)[:,:24,:]
            elif data.shape[1] == 266:
                print(data.shape)
                joints_num = 22
                data = torch.from_numpy(data).float()
                data = data[:1024,:266]
                joint3d = recover_from_ric266v_grad(data, joints_num)
                # joint3d = recover_from_ric266(data, joints_num)
            elif data.shape[1] == 338:
                # print(data.shape)
                joints_num = 22
                data = torch.from_numpy(data).float()
                data = data[:1024,:338]
                joints = recover_from_smplx_v(data, joints_num)
                joint3d = joints[:,:24,:]
            # print(data.shape)
        else:
            raise
        print(pkl)

        joint3d = joint3d[:1024,:22,:]      # Attention
        assert len(joint3d.shape) == 3
        joint3d = joint3d.reshape(joint3d.shape[0], 22*3).detach().cpu().numpy()
        print('joint3d', joint3d.shape)
            
        
        # print(extract_manual_features(joint3d.reshape(-1, 24, 3)))
        roott = joint3d[:1, :3]  # the root Tx72 (Tx(24x3))
        # print(roott)
        joint3d = joint3d - np.tile(roott, (1, 22))  # Calculate relative offset with respect to root

        # relative
        joint3d_relative = joint3d.copy()
        joint3d_relative = joint3d_relative.reshape(-1, 22, 3)
        joint3d_relative[:, 1:, :] = joint3d_relative[:, 1:, :] - joint3d_relative[:, 0:1, :]
        np.save(os.path.join(root, 'kinetic_features', pkl), extract_kinetic_features(joint3d_relative.reshape(-1, 22, 3)))
        np.save(os.path.join(root, 'manual_features_new', pkl), extract_manual_features(joint3d_relative.reshape(-1, 22, 3)))

        # np.save(os.path.join(root, 'kinetic_features', pkl), extract_kinetic_features(joint3d.reshape(-1, 22, 3)))
        # np.save(os.path.join(root, 'manual_features_new', pkl), extract_manual_features(joint3d.reshape(-1, 22, 3)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modir", type=str, default='None', help="the pred motion root"
    ) 
    opt = parser.parse_args()
    device = f"cuda:1"
    smplx_model = SMPLX_Skeleton()
    mod = '_relative'
    mod = '_global'

    # gt_root = '/data2/lrh/dataset/fine_dance/gound/div_by_time/mofea319_256/'
    # gt_root = '/data2/lrh/dataset/fine_dance/gound/mofea319/'
    # gt_root = '/data2/lrh/dataset/fine_dance/div_by_time/motion_fea319_256/'
    # gt_root = '/data2/lrh/dataset/fine_dance/div_by_time/motion_fea319_1024/'
    # gt_root = '/data2/lrh/dataset/fine_dance/gound/div_by_time/mofea319_1024/'
    # gt_root = '/data2/lrh/dataset/fine_dance/gound/div_by_time/mofea319_1800/'
    # gt_root = 'experiments/compare/gt_finedance/gound_1024_test_relative'

    gt_root = '/data2/lrh/dataset/fine_dance/gound/mofea319'
    # gt_root = '/data/lrh/datasets/fine_dance/gound/mofea319/'
    # pred_root = '/data2/lrh/dataset/fine_dance/gound/smplx_mofea266/new_joint_vecs_test'
    # gt_root = '/data2/lrh/dataset/fine_dance/gound/newmofea266/new_joint_vecs'
    # pred_root = '/data2/lrh/project/dance/LodgePlus/Lodge_plus_exp2/DanceDiffusion/experimentsFD/FineDance0701_266/Local_Module/velboneloss_diff_bc768/0702/autore'
    pred_root = '/data2/lrh/project/dance/LodgePlus/Lodge_plus_smpl/experimentsFD139/vqvae/FINEDANCE_139CUT/0708_win128_Norm_bc1024/eval_flat/eval_motion_test'
    # pred_root = '/data2/lrh/project/dance/LodgePlus/Lodge_plus_exp/DanceDiffusn/experimentsFD/Local_Module/FineDance0624_266Replacenoroot_TruOri_diff_bc768/0627/inpaint_key/gpt'

    
    # pred_root = '/data/lrh/project/dance/LongV2/LongV2Exp1/abandon/infer098_64-full-debug/gpr'
    # pred_root = '/data2/lrh/project/dance/LodgePlus/Lodge_plus_exp/abandon/finedance/gttest'
    # pred_root = '/data/lrh/project/dance/LongV2/LongV2Exp1/abandon/jay_output/gpt_diffu'

    # pred_root = '/data2/lrh/project/dance/LodgePlus/Lodge_plus_exp/DanceDiffusion/experiments/Local_Module/AIST0613_replace_key_Norm_loss266origin_diff_bc768/res_inpaint_key_modiftatx0'
    # pred_root = '/data2/lrh/project/dance/LodgePlus/Lodge_plus_exp/DanceDiffusion/experimentsFD/Local_Module/FineDance0621_266lossReplaceTruOri_diff_bc768/0624/replaceonlynoise_inpaint_key'
    # pred_root = '/data2/lrh/project/dance/LodgePlus/Lodge_plus_exp/DanceDiffusion/experimentsFD/Local_Module/FineDance0621_266lossReplaceTruOri_diff_bc768/0624/replaceonlynoise_inpaint_key/gpt'
    # pred_root = '/data2/lrh/project/dance/LodgePlus/40/LongV2Exp1/abandon/infer098_64-full-debug/gpr'
    print('Calculating and saving features')



    if opt.modir != 'None':
        pred_root = opt.modir
    # calc_and_save_feats(gt_root)
    calc_and_save_feats(pred_root)
    
    print('Calculating metrics')
    print("gt_root", gt_root)
    print("pred_root", pred_root)
    print(quantized_metrics(pred_root, gt_root))
    print("gt_root", gt_root)
    print("pred_root", pred_root)
