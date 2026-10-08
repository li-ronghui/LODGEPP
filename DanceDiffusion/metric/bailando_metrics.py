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
from smplx import SMPL
# from render import ax_to_6v
sys.path.append(os.getcwd())
from dld.data.utils.smplfk import SMPLSkeleton, do_smplxfk,ax_from_6v
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
    # if gt_freatures_k.shape[1] == 72:
    #     gt_freatures_k = gt_freatures_k[:,:66]
    # if pred_features_k.shape[1] == 66:
    #     gt_freatures_k = gt_freatures_k[:,:66]

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
    
    test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211", "193", "179", "065", "137", "161", "092",  "037", "109", "204", "144"]
    for pkl in tqdm(os.listdir(root)):
        if os.path.isdir(os.path.join(root, pkl)):
            continue
        # if pkl[:3] == 'dod':
        #     continue
        # if pkl[:3] not in test_list:
        #     continue
        if pkl[-3:] == 'pkl':
            data = pickle.load(open(os.path.join(root, pkl), "rb"))
            print(data.keys())
            model_q = torch.from_numpy(data['smpl_poses'] )
            model_x = torch.from_numpy(data['smpl_trans'] )
            print("model_q", model_q.shape)
            print("model_x", model_x.shape)
            # model_q156
            # with torch.no_grad():
            #     joint3d = smpl_model.forward(model_q156.unsqueeze(0), model_x.unsqueeze(0)).squeeze(0)[:,:24,:]
            #     print("joint3d", joint3d.shape)
            smpl_poses = torch.cat( [model_q[:,3:], torch.zeros(model_q.shape[0], 6) ],dim=1)

            joint3d = smpl.forward(
                    global_orient=(model_q[:,:3]).float(),
                    body_pose=(smpl_poses).float(),
                    transl=(model_x).float(),
                    ).joints[:, 0:24, :]
            print('joint3d', joint3d.shape)
            
        elif pkl[-3:] == 'npy':
            data = np.load(os.path.join(root, pkl))
            if len(data.shape) == 2:
                if data.shape[1] not in  [266,290]:
                    if data.shape[1] == 139:
                        data = data[:,:139]
                        # data = torch.from_numpy(data).to(device)   
                        trans = torch.from_numpy(data[:,4:7])
                        poses6d = data[:,7:]
                        axis = poses6d.reshape(data.shape[0], -1, 6)
                        axis = ax_from_6v(torch.from_numpy(axis))       #.detach().cpu().numpy()
                        print("axis", axis.shape)
                        axis = axis.reshape(data.shape[0], 22*3)
                        global_orient = axis[:,:3]
                        # body_pose = np.concatenate([axis[:,3:], np.zeros([data.shape[0], 6])], axis=1)
                        
                        body_pose = torch.cat( [axis[:,3:], torch.zeros(axis.shape[0], 6) ],dim=1)

                    elif data.shape[1] == 135:
                        data = data[:,:135]
                        # data = torch.from_numpy(data).to(device)   
                        # data = torch.cat([torch.zeros([data.shape[0], 4]).to(data),  data], dim=1) 

                        trans = torch.from_numpy(data[:,:3])
                        poses6d = data[:,3:]
                        axis = poses6d.reshape(data.shape[0], -1, 6)
                        axis = ax_from_6v(torch.from_numpy(axis))       #.detach().cpu().numpy()
                        print("axis", axis.shape)
                        axis = axis.reshape(data.shape[0], 22*3)
                        global_orient = axis[:,:3]
                        
                        body_pose = torch.cat( [axis[:,3:], torch.zeros(axis.shape[0], 6) ],dim=1)

                    # print(data.shape)
                    elif data.shape[1] == 151:
                        # data = torch.from_numpy(data).to(device)   
                        trans = torch.from_numpy(data[:,4:7])
                        poses6d = data[:,7:]
                        axis = poses6d.reshape(data.shape[0], 24, 6)
                        axis = ax_from_6v(torch.from_numpy(axis))       #.detach().cpu().numpy()
                        print("axis", axis.shape)
                        axis = axis.reshape(data.shape[0], 24*3)
                        global_orient = axis[:,:3].clone()
                        # body_pose = np.concatenate([axis[:,3:], np.zeros([data.shape[0], 6])], axis=1)
                        
                        body_pose = axis[:,3:].clone()     # torch.cat( [axis[:,3:], torch.zeros(axis.shape[0], 6) ],dim=1)
                    
                    with torch.no_grad():
                        # joint3d = do_smplxfk(data, smplx_model)[:,:24,:]
                        # joint3d = smpl_model.forward(axis.unsqueeze(0), trans.unsqueeze(0)).squeeze(0)[:,:24,:]
                        joint3d = smpl.forward(
                            global_orient=(global_orient).float(),
                            body_pose=(body_pose).float(),
                            transl=(trans).float(),
                            ).joints[:, 0:24, :].detach().cpu().numpy()
                        print('joint3d', joint3d.shape)

                elif data.shape[1] == 266:
                    joint3d = recover_from_ric266v_grad(torch.from_numpy(data), 22).detach().cpu().numpy()
                elif data.shape[1] == 290:
                    # joint3d = recover_from_ricv_norotinit_grad(torch.from_numpy(data), 24).detach().cpu().numpy()
                    joint3d = recover_from_ric266(torch.from_numpy(data), 24).detach().cpu().numpy()
            elif len(data.shape) == 3:
                joint3d = data  #torch.from_numpy(data)
            joinstnum = joint3d.shape[1]
        else:
            continue
        print(pkl)
        joint3d = joint3d[:600,:24,:]      # Attention
        assert len(joint3d.shape) == 3
        if joinstnum == 24:
            joint3d = joint3d.reshape(joint3d.shape[0], 24*3)#.detach().cpu().numpy()
        elif joinstnum == 22:
            joint3d = joint3d.reshape(joint3d.shape[0], 22*3)#.detach().cpu().numpy()
            
        
        # print(extract_manual_features(joint3d.reshape(-1, 24, 3)))
        roott = joint3d[:1, :3]  # the root Tx72 (Tx(24x3))
        # print(roott)
        if joinstnum == 24:
            joint3d = joint3d - np.tile(roott, (1, 24))  # Calculate relative offset with respect to root

            np.save(os.path.join(root, 'kinetic_features', pkl), extract_kinetic_features(joint3d.reshape(-1, 24, 3)))
            np.save(os.path.join(root, 'manual_features_new', pkl), extract_manual_features(joint3d.reshape(-1, 24, 3)))
        elif joinstnum == 22:
            joint3d = joint3d - np.tile(roott, (1, 22))  # Calculate relative offset with respect to root
            np.save(os.path.join(root, 'kinetic_features', pkl), extract_kinetic_features(joint3d.reshape(-1, 22, 3)))
            np.save(os.path.join(root, 'manual_features_new', pkl), extract_manual_features(joint3d.reshape(-1, 22, 3)))

        # relative
        # joint3d_relative = joint3d.copy()
        # joint3d_relative = joint3d_relative.reshape(-1, 24, 3)
        # joint3d_relative[:, 1:, :] = joint3d_relative[:, 1:, :] - joint3d_relative[:, 0:1, :]
        # np.save(os.path.join(root, 'kinetic_features', pkl), extract_kinetic_features(joint3d_relative.reshape(-1, 24, 3)))
        # np.save(os.path.join(root, 'manual_features_new', pkl), extract_manual_features(joint3d_relative.reshape(-1, 24, 3)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--modir", type=str, default='None', help="the pred motion root"
    ) 
    opt = parser.parse_args()
    device = f"cuda:1"
    # smpl_model = SMPLSkeleton()
    smpl = SMPL(model_path='/data2/lrh/project/dance/Bailando/smpl', gender='MALE', batch_size=1)

    mod = '_relative'
    mod = '_global'
    
    # gt_root = '/data2/lrh/dataset/aist/data/origin/60fps/fullset/mofea266/new_joint_vecs'
    # pred_root = '/data2/lrh/project/dance/LodgePlus/Lodge_plus_exp/DanceDiffusion/experiments/Local_Module/AIST0613_replace_key_Norm_loss266origin_diff_bc768/res_inpaint_key_modiftatx0/twice'
    # pred_root = '/data2/lrh/project/dance/LodgePlus/Lodge_plus_exp/DanceDiffusion/experiments/Local_Module/AIST0613_replace_key_Norm_loss266origin_diff_bc768/gpt/twice'


    # gt_root = '/data2/lrh/project/dance/Bailando/data/aist_features_zero_start'
    # pred_root = '/data2/lrh/project/dance/Lodge/lodge302/experiments/Local_Module/AISTPP_relative_Norm_128len_139/samples_dod_2499_499_inpaint_soft_ddim_notranscontrol_2024-03-14-08-38-58/concat/twice'
    # pred_root = '/data2/lrh/dataset/aist/data/followB/30fps/fullset/mofea290_ori/new_joint_vecs_test/twice'
    # pred_root = '/data2/lrh/project/dance/LodgePlus/Lodge_plus_exp/DanceDiffusion/experiments/Local_Module/AIST0613_replace_key_Norm_loss266origin_diff_bc768/res_inpaint_key_modiftatx0/twice'
    # pred_root = '/data2/lrh/dataset/aist/data/followB/60fps/fullset/joints_test'
    # pred_root = '/data2/lrh/dataset/aist/data/followB/60fps/fullset/mofea290_ori/new_joint_vecs_test'
    # GT
    # pred_root = '/data2/lrh/project/dance/Bailando/data/aist_features_zero_start_test_60fps'
    # pred_root =  '/data2/lrh/dataset/aist/data/origin/60fps/fullset/mofea266/new_joint_vecs_testdir'
    # gt_root = '/data2/lrh/dataset/aist/data/followB/30fps/fullset/mofea266/new_joint_vecs'
    # pred_root = '/data2/lrh/dataset/aist/data/followB/30fps/fullset/mofea266/new_joint_vecs_testset'

    # gt_root = '/data2/lrh/dataset/aist/data/followB/30fps/fullset/mofea266/new_joint_vecs'
    # pred_root = '/data2/lrh/dataset/aist/data/followB/60fps/fullset/mofea290_ori/new_joint_vecs_test'
    # dod
    # pred_root = '/data2/lrh/project/dance/Lodge/lodge_pylight/experiments/Edge_Module/AIST0125_noNorm_128len_139_diff_bc768/samples_dod_2024-01-26-02-51-33/concat/twice/'
    # dod v2
    # pred_root = '/data2/lrh/project/dance/Lodge/lodge_pylight/experiments/Edge_Module/AIST0125_noNorm_128len_139_diff_bc768/samples_dod_2024-01-25-22-39-30/concat/twice' 
    # # 不加soft guidance
    # pred_root = '/data2/lrh/project/dance/Lodge/lodge_pylight/experiments/Edge_Module/AIST0125_noNorm_128len_139_diff_bc768/samples_dod_nosoft_2024-01-27-02-13-40/concat/twice/'
    # # edge long
    # pred_root = '/data2/lrh/project/dance/Lodge/lodge_pylight/experiments/Edge_Module/AIST0125_noNorm_128len_139_diff_bc768/samples_long_2024-01-27-02-51-10-twice' 


    # gt_root = '/data2/lrh/project/dance/Bailando/data/aist_features_zero_start_relative'
    # pred_root = '/data2/lrh/project/dance/Lodge/lodge_pylight/experiments/Edge_Module/AIST60FPS_0126_noNorm_128len_139/samples_dod_2024-01-27-00-37-01/concat/npy/relative_feature'
    # 最新测试  # 2024
    gt_root = '/data2/lrh/project/dance/Bailando/data/aist_features_zero_start_30fps'
    pred_root = '/data2/lrh/project/dance/LodgePlus/Lodge_plus_smpl/DanceDiffusion/experimentsAIST151/0712/Local_Module/replace8_win3_len128_relativeloss_Norm/0716_dhf/beatguide_x6'
    # pred_root = '/data2/lrh/dataset/aist/data/followB/30fps/fullset/mofea290_ori/new_joint_vecs_test'
    # pred_root = '/data2/lrh/project/dance/LodgePlus/Lodge_plus_exp/DanceDiffusion/experiments290/Local_Module/AIST0616_Norm_win2_loss290noFFT_diff_bc768/autoregress'
    # pred_root = '/data2/lrh/project/dance/LodgePlus/Lodge_plus_exp/DanceDiffusion/experiments290/Local_Module/AIST0620_win2_loss290replaceTruori_diff_bc768/replaceonlynoise_inpaint_key'
    # pred_root = '/data2/lrh/project/dance/LodgePlus/Lodge_plus_exp/DanceDiffusion/experiments290/Local_Module/AIST0620_win2_loss290replaceTruori_diff_bc768/0622/replaceonlynoiseinpaint_key_withkeymotion'

    # pred_root = '/data2/lrh/project/dance/LodgePlus/Lodge_plus_exp/DanceDiffusion/experiments/Local_Module/AIST0613_replace_key_Norm_loss266origin_diff_bc768/gpt/'
    # pred_root = '/data2/lrh/project/dance/Bailando/data/aist_features_zero_start_test_30fps'
    # pred_root = '/data2/lrh/project/dance/Lodge/lodge_pylight/experiments/Edge_Module/AIST0125_noNorm_128len_139_diff_bc768/samples_dod_2024-01-26-02-51-33/concat/npy/bailando_fea/'
    # pred_root = '/data2/lrh/project/dance/Lodge/lodge_pylight/experiments/Edge_Module/AIST0125_noNorm_128len_139_diff_bc768/samples_dod_2024-01-26-02-51-33/concat/npy/'
    # 不加soft guidance
    # pred_root = '/data2/lrh/project/dance/Lodge/lodge_pylight/experiments/Edge_Module/AIST0125_noNorm_128len_139_diff_bc768/samples_dod_nosoft_2024-01-27-02-13-40/concat/npy'
    # edge long
    # pred_root = '/data2/lrh/project/dance/Lodge/lodge_pylight/experiments/Edge_Module/AIST0125_noNorm_128len_139_diff_bc768/samples_long_2024-01-27-02-51-10' 

    # pred_root = '/data2/lrh/project/dance/Lodge/lodge_pylight/experiments/Edge_Module/AIST0125_noNorm_128len_139_diff_bc768/samples_dod_2024-01-25-22-39-30/concat/npy'
    # gt_root = '/data2/lrh/project/dance/Bailando/data/aist_features_zero_start_30fps_relative'
    # pred_root = '/data2/lrh/project/dance/Bailando/data/aist_features_zero_start_test_30fps_relative'
    # pred_root = '/data2/lrh/project/dance/Lodge/lodge_pylight/experiments/Edge_Module/AIST0125_noNorm_128len_139_diff_bc768/samples_dod_2024-01-26-02-51-33/concat/npy'
    # pred_root = '/data2/lrh/project/dance/Lodge/lodge_pylight/experiments/Edge_Module/AIST0125_relative_noNorm_128len_139_diff_bc768/samples_dod_2024-01-26-06-43-09/concat/npy'
    
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
