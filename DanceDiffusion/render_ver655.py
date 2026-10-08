'''
render double dance in mesh
'''
import os

from matplotlib import axis
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
from re import split
import subprocess
import sys
import cv2
from smplx import SMPL, SMPLH, SMPLX
from tqdm import tqdm
import torch, pickle
from dld.data.utils.motion_process import recover_from_ric, recover_from_ric266, recover_from_ric266v, recover_from_smplx_v, recover_from_smplx, recover_from_smplx_globalinit_v
import numpy as np
from dld.data.utils.smplfk import SMPLX_Skeleton, do_smplxfk
import argparse
from pytorch3d.transforms import (axis_angle_to_matrix, matrix_to_axis_angle,
                                  matrix_to_quaternion, matrix_to_rotation_6d,
                                  quaternion_to_matrix, rotation_6d_to_matrix)
import pyrender
import trimesh


def quat_to_6v(q):
    assert q.shape[-1] == 4
    mat = quaternion_to_matrix(q)
    mat = matrix_to_rotation_6d(mat)
    return mat
def quat_from_6v(q):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    quat = matrix_to_quaternion(mat)
    return quat
def ax_to_6v(q):
    assert q.shape[-1] == 3
    mat = axis_angle_to_matrix(q)
    mat = matrix_to_rotation_6d(mat)
    return mat
def ax_from_6v(q):
    assert q.shape[-1] == 6
    mat = rotation_6d_to_matrix(q)
    ax = matrix_to_axis_angle(mat)
    return ax

def get_joints_from_file(file,use_rela=False):
    rela_dis=None
    if file[-3:] == 'pkl':
        pkl_data = pickle.load(open(os.path.join(modir, file), 'rb'))
        # file_li.append(os.path.join(outdir, file.replace('pkl', 'gif')))
        smpl_poses = torch.from_numpy(pkl_data["smpl_poses"]).reshape(-1, 22, 3)
        smpl_trans = torch.from_numpy(pkl_data["smpl_trans"])
        smpl_poses = ax_to_6v(smpl_poses).view(smpl_trans.shape[0], 132)
        data139 = torch.cat([torch.zeros(smpl_poses.shape[0], 4).to(smpl_poses), smpl_trans, smpl_poses], dim=-1)
        print("data139", data139.shape)
        joints = do_smplxfk(data139, smplx)[:, :22, :]
    elif file[-3:] == 'npy':
        motion = np.load(os.path.join(modir, file))  # [30:150]
        # file_li.append(os.path.join(outdir, file.replace('npy', 'gif')))
        print("motion shape", motion.shape)
        if motion.shape[1] == 263 or motion.shape[1] == 266:
            # motion = (motion * std) + mean
            motion = torch.from_numpy(motion)
        elif len(motion.shape) == 3 and motion.shape[-1] == 263:
            motion = torch.from_numpy(motion)[8]
        if motion.shape[1] == 263:
            joints = recover_from_ric(motion.to(dtype=torch.float), joints_num)  # .detach().cpu().numpy()
        elif motion.shape[1] == 266:
            joints = recover_from_ric266(motion.to(dtype=torch.float), joints_num)
            print("in 266")
            print("joints  11", joints[0,:3])
            print("joints  11", joints.shape)
        elif motion.shape[1] == 338:
            motion = torch.from_numpy(motion)
            # joints = recover_from_smplx_v(motion[:,:338].to(dtype=torch.float), joints_num)
            joints = recover_from_smplx_globalinit_v(motion[:,:338].to(dtype=torch.float), joints_num)
            print('joints', joints.shape)
        elif motion.shape[1] == 340:
            motion = torch.from_numpy(motion)
            joints = recover_from_smplx_v(motion[:, :338].to(dtype=torch.float), joints_num)
            rela_dis = np.pad(motion[:,338:], ((0, 0), (0, 1)), mode='constant', constant_values=0)
        elif motion.shape[1] > 338:
            motion = torch.from_numpy(motion)
            joints = recover_from_smplx_globalinit_v(motion.to(dtype=torch.float), joints_num=710)
            print('joints', joints.shape)
        else:
            print("motion shape", motion.shape)
            raise("error of motion shape")

    # joints = joints.reshape(joints.shape[0], joints_num * 3).detach().cpu().numpy()
    # roott = joints[:1, :3]  # the root Tx72 (Tx(24x3))
    # joints = joints - np.tile(roott, (1, joints_num))
    joints = joints.reshape(-1, joints_num, 3)
    if use_rela:
        joints = np.concatenate((joints, rela_dis[:, np.newaxis, :]), axis=1)  # T,56,3

    # joints = torch.from_numpy(joints)

    return joints

def look_at(eye, center, up):
    front = eye - center
    front = front / np.linalg.norm(front)
    right = np.cross(up, front)
    right = right / np.linalg.norm(right)
    up_new = np.cross(front, right)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = np.stack([right, up_new, front]).transpose()
    camera_pose[:3, 3] = eye
    return camera_pose


class MovieMaker():
    def __init__(self, save_path) -> None:

        self.mag = 2
        self.eyes = np.array([[3, -3, 2], [0, 0, -2], [0, 0, 4], [-8, -8, 1], [0, -2, 4], [0, 3, 5]])       # 双人舞设置相机更远一些
        self.centers = np.array([[0, 0, 0], [0, 0, 0], [0, 0.5, 0], [0, 0, -1], [0, 0.5, 0], [0, 0.5, 0]])
        self.ups = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0], [0, 0, -1], [0, 1, 0], [0, 1, 0]])
        self.save_path = save_path

        self.fps = args.fps
        self.img_size = (1200, 1200)

        ipadd = "10.103.11.45"  # getip()
        if ipadd == "10.103.11.45":
            SMPLH_path = "/data/human/datasets/smpl_model/smplh/SMPLH_MALE.pkl"
            SMPL_path = "/data/human/datasets/smpl_model/smpl/SMPL_MALE.pkl"
            SMPLX_path = "/data/human/datasets/smpl_model/smplx/SMPLX_NEUTRAL.npz"
            trimesh_path = '/data2/lrh/floor/NORMAL_new.obj'
            self.faces655 = np.load('/data2/lrh/dataset/fine_dance/RepairedDouble/beforesplit/vertices/faces655.npy')
        elif ipadd == "10.103.11.40":
            SMPLH_path = "/home/nfs/lrh/smpl_model/smplh/SMPLH_MALE.pkl"
            SMPL_path = "/home/nfs/lrh/smpl_model/smpl/SMPL_MALE.pkl"
            SMPLX_path = "/home/nfs/lrh/smpl_model/smplx/SMPLX_NEUTRAL.npz"
            trimesh_path = '/home/nfs/zyl/asset/NORMAL_new.obj'
        else:
            raise ("error of machine ip")

        self.smplh = SMPLH(SMPLH_path, use_pca=False, flat_hand_mean=False)
        self.smplh.to(f'cuda:{args.gpu}').eval()

        self.smpl = SMPL(SMPL_path)
        self.smpl.to(f'cuda:{args.gpu}').eval()

        self.smplx = SMPLX(SMPLX_path, use_pca=False, flat_hand_mean=False).eval()
        self.smplx.to(f'cuda:{args.gpu}').eval()

        self.scene = pyrender.Scene()
        self.mesh = trimesh.load(trimesh_path)
        # tile_color = [0.8, 0.8, 0.8, 1.0]  # RGBA格式，这里表示瓷砖的颜色
        # material_tile = pyrender.MetallicRoughnessMaterial(baseColorFactor=tile_color, metallicFactor=0.2,
        #                                                    roughnessFactor=0.2)
        floor_mesh = pyrender.Mesh.from_trimesh(self.mesh)
        self.scene.add(floor_mesh)
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = look_at(self.eyes[5], self.centers[5], self.ups[5])  # 2
        self.scene.add(camera, pose=camera_pose)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
        self.scene.add(light, pose=camera_pose)
        self.r = pyrender.OffscreenRenderer(self.img_size[0], self.img_size[1])


    def save_video(self, save_path, color_list):
        # save_path = os.path.join(save_path,'move.mp4')
        f = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videowriter = cv2.VideoWriter(save_path, f, self.fps, self.img_size)
        for i in range(len(color_list)):
            videowriter.write(color_list[i][:, :, ::-1])
        videowriter.release()
    def save_two_video(self, save_path, color_list0,color_list1):
        # save_path = os.path.join(save_path,'move.mp4')
        f = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videowriter = cv2.VideoWriter(save_path, f, self.fps, self.img_size)
        for i in range(len(color_list0)):
            videowriter.write(color_list0[i][:, :, ::-1])
        for i in range(len(color_list1)):
            videowriter.write(color_list1[i][:, :, ::-1])
        videowriter.release()
    def get_imgs(self, motion):
        meshes = self.motion2mesh(motion)
        imgs = self.render_imgs(meshes)
        return np.concatenate(imgs, axis=1)

    def motion2mesh(self, motion):
        if args.mode == "smpl":
            output = self.smpl.forward(
                betas=torch.zeros([motion.shape[0], 10]).to(motion.device),
                transl=motion[:, :3],
                global_orient=motion[:, 3:6],
                body_pose=torch.cat([motion[:, 6:69], motion[:, 69:72], motion[:, 114:117]], dim=1)
            )
        elif args.mode == "smplh":
            output = self.smplh.forward(
                betas=torch.zeros([motion.shape[0], 10]).to(motion.device),
                # transl = motion[:,:3],
                transl=torch.tensor([[0, 0, -1]]).expand(motion.shape[0], -1).to(motion.device),
                global_orient=motion[:, 3:6],
                body_pose=motion[:, 6:69],
                left_hand_pose=motion[:, 69:114],
                right_hand_pose=motion[:, 114:159],
            )
        elif args.mode == "smplx":
            setbetas = torch.zeros()([motion.shape[0], 10]).to(motion.device)
            setbetas[:, 1] = -1.2
            output = self.smplx.forward(
                betas=setbetas,
                # transl = motion[:,:3],
                transl=motion[:, :3],
                global_orient=motion[:, 3:6],
                body_pose=motion[:, 6:69],
                jaw_pose=torch.zeros([motion.shape[0], 3]).to(motion),
                leye_pose=torch.zeros([motion.shape[0], 3]).to(motion),
                reye_pose=torch.zeros([motion.shape[0], 3]).to(motion),
                left_hand_pose = motion[:,69:69+15*3],
                right_hand_pose = motion[:,-45:],
                expression=torch.zeros([motion.shape[0], 10]).to(motion),
            )
        # tespo
        meshes = []
        for i in range(output.vertices.shape[0]):
            if args.mode == 'smplh':
                mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smplh.faces)
            elif args.mode == 'smplx':
                mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smplx.faces)
            elif args.mode == 'smpl':
                mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smpl.faces)
            # mesh.export(os.path.join(self.save_path, f'{i}.obj'))
            meshes.append(mesh)

        return meshes

    def render_multi_view(self, meshes, music_file, tab='', eyes=None, centers=None, ups=None, views=1):
        if eyes and centers and ups:
            assert eyes.shape == centers.shape == ups.shape
        else:
            eyes = self.eyes
            centers = self.centers
            ups = self.ups

        for i in range(views):
            color_list = self.render_single_view(meshes, eyes[1], centers[1], ups[1])
            movie_file = os.path.join(self.save_path, tab + '-' + str(i) + '.mp4')
            output_file = os.path.join(self.save_path, tab + '-' + str(i) + '-music.mp4')
            self.save_video(movie_file, color_list)
            if music_file is not None:
                subprocess.run(
                    ['/home/lrh/Documents/ffmpeg-6.0-amd64-static/ffmpeg', '-i', movie_file, '-i', music_file,
                     '-shortest', output_file])
            else:
                subprocess.run(['/home/lrh/Documents/ffmpeg-6.0-amd64-static/ffmpeg', '-i', movie_file, output_file])
            os.remove(movie_file)
    def render_two_person(self,meshes0,meshes1):
        # print("frames",len(meshes0),len(meshes0[0]))
        num = min(len(meshes0),len(meshes1))
        meshes0 = meshes0[:num]
        print('--meshes0--', len(meshes0))
        meshes1 = meshes1[:num]
        print('--meshes1--', len(meshes1))
        red_color =  [255,192,203,1.000]   # [1.0, 0.0, 0.0, 1.0]  # RGBA格式，这里表示红色    小粉
        red_color = [x/255 for x in red_color]
        material_red = pyrender.MetallicRoughnessMaterial(baseColorFactor=red_color)
        blue_color = [127,255,212,1.000]     # [0.0, 0.0, 1.0, 1.0]  # RGBA格式，这里表示蓝色  宝蓝色
        blue_color = [x/255 for x in blue_color]
        material_blue = pyrender.MetallicRoughnessMaterial(baseColorFactor=blue_color)
        color_list = []
        for i in tqdm(range(num)):
            mesh_nodes = []
            for mesh in meshes0[i]:
                render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material_red)
                mesh_node = self.scene.add(render_mesh)
                mesh_nodes.append(mesh_node)
            for mesh in meshes1[i]:
                render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material_blue)
                mesh_node = self.scene.add(render_mesh)
                mesh_nodes.append(mesh_node)
            
            color, _ = self.r.render(self.scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
            color = color.copy()
            color_list.append(color)
            for mesh_node in mesh_nodes:
                self.scene.remove_node(mesh_node)
        return color_list
    def render_single_view(self, meshes):
        num = len(meshes)
        color_list = []
        for i in tqdm(range(num)):
            mesh_nodes = []
            for mesh in meshes[i]:
                render_mesh = pyrender.Mesh.from_trimesh(mesh)
                mesh_node = self.scene.add(render_mesh)
                mesh_nodes.append(mesh_node)
            color, _ = self.r.render(self.scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
            color = color.copy()
            color_list.append(color)
            for mesh_node in mesh_nodes:
                self.scene.remove_node(mesh_node)
        return color_list

    def render_imgs(self, meshes):
        colors = []
        for mesh in meshes:
            render_mesh = pyrender.Mesh.from_trimesh(mesh)
            mesh_node = self.scene.add(render_mesh)
            color, _ = self.r.render(self.scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL)
            colors.append(color)
            self.scene.remove_node(mesh_node)

        return colors
        # cv2.imwrite(os.path.join(self.save_path, 'test.jpg'), color[:,:,::-1])

    def run_two(self,seq_rot0, seq_rot1, setbetas0, setbetas1,  music_file=None, dance_name='000',tab0='',tab1='', save_pt=False):
        meshes0=self.get_out_from_seq(seq_rot0, setbetas0, tab=tab0,save_pt=save_pt)
        meshes1=self.get_out_from_seq(seq_rot1, setbetas1, tab=tab1, save_pt=save_pt)
        color_list = self.render_two_person(meshes0,meshes1)
        movie_file = os.path.join(self.save_path, dance_name + 'tmp.mp4')
        output_file = os.path.join(self.save_path, dance_name + 'z.mp4')
        self.save_video(movie_file, color_list)
        if music_file is not None:
            subprocess.run(
                ['/home/lrh/Documents/ffmpeg-6.0-amd64-static/ffmpeg', '-i', movie_file, '-i', music_file,
                 '-shortest',
                 output_file])
        else:

            subprocess.run(
                ['/home/lrh/Documents/ffmpeg-6.0-amd64-static/ffmpeg', '-i', movie_file, output_file])
        os.remove(movie_file)

    def get_out_from_seq(self, seq_rot, setbetas, tab='', save_pt=False):
        if isinstance(seq_rot, np.ndarray):
            seq_rot = torch.tensor(seq_rot, dtype=torch.float32, device=f'cuda:{args.gpu}')
        if save_pt:
            torch.save(seq_rot.detach().cpu(), os.path.join(self.save_path, tab + '_pose.pt'))

        if len(seq_rot.shape) == 2 and args.mode[:4] == 'smpl':
            B, D = seq_rot.shape
            if setbetas is not None:
                setbetas = torch.from_numpy(setbetas).to(seq_rot)
            output=None
            if args.mode == "smpl":
                print("using smpl!!!")
                output = self.smpl.forward(
                    betas=setbetas.unsqueeze(0).repeat(seq_rot.shape[0], 1),
                    transl=seq_rot[:, :3],
                    global_orient=seq_rot[:, 3:6],
                    body_pose=torch.cat([seq_rot[:, 6:69], seq_rot[:, 69:72], seq_rot[:, 114:117]], dim=1)
                )

            elif args.mode == "smplh":
                print("using smplh!!!")
                output = self.smplh.forward(
                    betas=setbetas.unsqueeze(0).repeat(seq_rot.shape[0], 1),
                    transl=seq_rot[:, :3],
                    global_orient=seq_rot[:, 3:6],
                    body_pose=seq_rot[:, 6:69],
                    left_hand_pose=seq_rot[:, 69:114],
                    right_hand_pose=seq_rot[:, 114:],  # torch.zeros([seq_rot.shape[0], 45]).to(seq_rot.device),
                    expression=torch.zeros([seq_rot.shape[0], 10]).to(seq_rot.device),
                )

            elif args.mode == "smplx":
                output = self.smplx.forward(
                    betas=setbetas.unsqueeze(0).repeat(seq_rot.shape[0], 1),
                    # transl = motion[:,:3],
                    transl=seq_rot[:, :3],
                    global_orient=seq_rot[:, 3:6],
                    body_pose=seq_rot[:, 6:69],
                    jaw_pose=seq_rot[:, 69:72],
                    leye_pose=seq_rot[:, 72:75],
                    reye_pose=seq_rot[:, 75:78],
                    left_hand_pose = seq_rot[:,78:78+45],
                    right_hand_pose = seq_rot[:,-45:],
                    expression=torch.zeros([seq_rot.shape[0], 10]).to(seq_rot),
                )
            N, V, DD = output.vertices.shape  # 150, 6890, 3
            vertices = output.vertices.reshape((B, -1, V, DD))  # # 150, 1, 6890, 3
        elif len(seq_rot.shape) == 3  and args.mode == 'ver655':       # 直接输入vertices
            B, _, _ = seq_rot.shape
            vertices = seq_rot

        meshes = []
        for i in range(B):
            # debug!
            # if int(i) > 10 and int(i) < 500:
            #     continue
            if not int(i) % 4 == 0:
                continue
            if int(i) > 60:
                break
            view = []
            if args.mode == 'smplh':
                mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smplh.faces)
            elif args.mode == 'smplx':
                mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smplx.faces)
            elif args.mode == 'smpl':
                mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smpl.faces)
            elif args.mode == 'ver655':
                mesh = trimesh.Trimesh(vertices[i].cpu(), self.faces655)
            view.append(mesh)
            meshes.append(view)
        return meshes
    def run(self, seq_rot, music_file=None, tab='', save_pt=False):
        if isinstance(seq_rot, np.ndarray):
            seq_rot = torch.tensor(seq_rot, dtype=torch.float32, device=f'cuda:{args.gpu}')

        if save_pt:
            torch.save(seq_rot.detach().cpu(), os.path.join(self.save_path, tab + '_pose.pt'))

        B, D = seq_rot.shape
        if args.mode == "smpl":
            print("using smpl!!!")
            output = self.smpl.forward(
                betas=torch.zeros([seq_rot.shape[0], 10]).to(seq_rot.device),
                transl=seq_rot[:, :3],
                global_orient=seq_rot[:, 3:6],
                body_pose=torch.cat([seq_rot[:, 6:69], seq_rot[:, 69:72], seq_rot[:, 114:117]], dim=1)
            )

        elif args.mode == "smplh":
            print("using smplh!!!")
            output = self.smplh.forward(
                betas=torch.zeros([seq_rot.shape[0], 10]).to(seq_rot.device),
                transl=seq_rot[:, :3],
                global_orient=seq_rot[:, 3:6],
                body_pose=seq_rot[:, 6:69],
                left_hand_pose=seq_rot[:, 69:114],
                # torch.zeros([seq_rot.shape[0], 45]).to(seq_rot.device),      # seq_rot[:,69:114],
                right_hand_pose=seq_rot[:, 114:],  # torch.zeros([seq_rot.shape[0], 45]).to(seq_rot.device),      #
                expression=torch.zeros([seq_rot.shape[0], 10]).to(seq_rot.device),
            )

        elif args.mode == "smplx":
            setbetas = torch.zeros([seq_rot.shape[0], 10]).to(seq_rot.device)
            setbetas[:, 1] = -1.2
            output = self.smplx.forward(
                betas=setbetas,
                # transl = motion[:,:3],
                transl=seq_rot[:, :3],
                global_orient=seq_rot[:, 3:6],
                body_pose=seq_rot[:, 6:69],
                jaw_pose=torch.zeros([seq_rot.shape[0], 3]).to(seq_rot),
                leye_pose=torch.zeros([seq_rot.shape[0], 3]).to(seq_rot),
                reye_pose=torch.zeros([seq_rot.shape[0], 3]).to(seq_rot),
                left_hand_pose=torch.zeros([seq_rot.shape[0], 45]).to(seq_rot),
                right_hand_pose=torch.zeros([seq_rot.shape[0], 45]).to(seq_rot),
                expression=torch.zeros([seq_rot.shape[0], 10]).to(seq_rot),
            )

        N, V, DD = output.vertices.shape  # 150, 6890, 3
        vertices = output.vertices.reshape((B, -1, V, DD))  # # 150, 1, 6890, 3

        meshes = []
        for i in range(B):
            # debug!
            # if int(i) > 10 and int(i) < 500:
            #     continue
            if not int(i) % 4 == 0:
                continue
            # if int(i) > 160:
            #     break
            view = []
            for v in vertices[i]:
                # vertices[:,2] *= -1
                if args.mode == 'smplh':
                    mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smplh.faces)
                elif args.mode == 'smplx':
                    mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smplx.faces)
                elif args.mode == 'smpl':
                    mesh = trimesh.Trimesh(output.vertices[i].cpu(), self.smpl.faces)
                elif args.mode == 'ver655':
                    mesh = trimesh.Trimesh(vertices[i].cpu(), self.faces655)
                # mesh.export(os.path.join(self.save_path, 'test.obj'))
                view.append(mesh)
            meshes.append(view)

        color_list = self.render_single_view(meshes)
        movie_file = os.path.join(self.save_path, tab + 'tmp.mp4')
        output_file = os.path.join(self.save_path, tab + 'z.mp4')
        self.save_video(movie_file, color_list)
        if music_file is not None:
            subprocess.run(
                ['/home/lrh/Documents/ffmpeg-6.0-amd64-static/ffmpeg', '-i', movie_file, '-i', music_file, '-shortest',
                 output_file])
        else:
            subprocess.run(['/home/lrh/Documents/ffmpeg-6.0-amd64-static/ffmpeg', '-i', movie_file, output_file])
        os.remove(movie_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="7")
    parser.add_argument("--modir", type=str, default="/home/lrh/Documents/dataset/RepairedDouble/new_xyz_vecs_128init")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--joints_num", type=int, default=710)   # 710
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--mode", type=str, default="ver655", choices=['smpl', 'smplh', 'smplx', "ver655"])

    # parser.add_argument("--song", type=str, default=None)
    args = parser.parse_args()
    print(args.gpu)
    joints_num = args.joints_num

    device = f'cuda:{args.gpu}'
    modir = args.modir
    if args.save_path is not None:
        outdir = args.save_path
    else:
        outdir = os.path.join(modir, 'video_from500_new')
    if not os.path.exists(outdir):
        os.makedirs(outdir)


    faces = np.load('/data2/lrh/dataset/fine_dance/RepairedDouble/beforesplit/vertices/faces655.npy')
    colors = np.ones((655, 4), dtype=np.float32) * 0.8

    for song in range(1, 100):
        songname = str(song).zfill(3)
        # file_a = os.path.join(modir, 'M'+songname + "_0@512@l0.npy")
        # print(file_a)
        # file_b = os.path.join(modir, 'M'+songname + "_1@512@l0.npy")
        file_a = os.path.join(modir, songname + "_1999_a.npy")
        file_b = os.path.join(modir, songname + "_1999_b.npy")
        # file_a = os.path.join(modir, songname + "_0@512@l0.pkl")
        # file_b = os.path.join(modir, songname + "_1@512@l0.pkl")

        if (not os.path.exists(file_a)) or  (not os.path.exists(file_b)):
            continue
        dance_name = songname + '.gif'
        
        if file_a.split('.')[-1] == 'npy':
            joints0 = get_joints_from_file(file_a,use_rela=False)[:, 55:]
            joints0 = joints0
            print('joints0', joints0.shape)
            joints1 = get_joints_from_file(file_b,use_rela=False)[:, 55:]
            joints1 = joints1
            print('joints1', joints1.shape)
            setbetas0 = None
            setbetas1 = None
        elif file_a.split('.')[-1] == 'pkl':
            with open(file_a, 'rb') as  f:
                data0 = pickle.load(f)
            joints0 = np.concatenate([ data0['trans'], data0['pose'].reshape(-1, 165) ], axis=1 )
            print('joints0---------', joints0[0,:3])
            setbetas0 =  data0['beta']


            with open(file_b, 'rb') as  f:
                data1 = pickle.load(f)
            joints1 = np.concatenate([ data1['trans'], data1['pose'].reshape(-1, 165) ], axis=1 )
            print('joints1---------', joints1[0,:3])
            setbetas1 =  data1['beta']

            # print('joints0', joints0[0])
            # print('joints1', joints1[0])

        visualizer = MovieMaker(save_path=outdir)
        visualizer.run_two(joints0, joints1, setbetas0, setbetas1,
                           dance_name=os.path.basename(file_b),
                           tab0=os.path.basename( songname + "_0").split(".")[0],
                           tab1=os.path.basename( songname + "_1").split(".")[0], 
                           music_file=None)
        

'''
渲染 smplx 文件，
--modir 'smplx pkl path' --mode smplx --fps 8 --outdir ./video_gt
渲染 ver655 文件，
--modir 'ver655 npy path' --mode ver655 --fps 8
'''




