import numpy as np
import os
import sys

music_dir = "/data2/lrh/dataset/aist/data/origin/30fps/fullset/musics35_30fps"
motion_dir = "/data2/lrh/dataset/aist/data/origin/60fps/fullset/mofea266/new_joint_vecs"

music_out = "/data2/lrh/dataset/aist/data/origin/30fps/fullset/div_bytime/musics35_30fps_"
motion_out = "/data2/lrh/dataset/aist/data/origin/30fps/fullset/div_bytime/mofea266/new_joint_vec_"

timelen = 128


music_out = music_out + str(timelen)
motion_out = motion_out + str(timelen)
if not os.path.exists(music_out):
    os.makedirs(music_out)
if not os.path.exists(motion_out):
    os.makedirs(motion_out)


for file in os.listdir(motion_dir):
    # if file[0] == 'M':
    #     continue
    if file[-3:] != 'npy':
        print(file[-3:])
        continue
    name = file.split(".")[0]
    music_fea = np.load(os.path.join(music_dir, file))
    motion_fea = np.load(os.path.join(motion_dir, file))
    # motion_feaM = np.load(os.path.join(motion_dir, 'M' + file))
    max_length = min(music_fea.shape[0], motion_fea.shape[0])

    iters = (max_length//timelen)
    max_length = iters*timelen
    music_fea = music_fea[:max_length, :]
    motion_fea = motion_fea[:max_length, :]
    # motion_feaM = motion_feaM[:max_length, :]

    for i in range(iters):
        music_clip = music_fea[i*timelen: (i+1)*timelen, :]
        motion_clip = motion_fea[i*timelen: (i+1)*timelen, :]
        # motion_clipM = motion_feaM[i*timelen: (i+1)*timelen, :]

        np.save(os.path.join(music_out, name + "z@" + str(i).zfill(3) + ".npy"), music_clip)
        np.save(os.path.join(motion_out, name + "z@" + str(i).zfill(3) + ".npy"), motion_clip)
        # np.save(os.path.join(motion_out, 'M' + name + "z@" + str(i).zfill(3) + ".npy"), motion_clip)
    