import numpy as np
import pickle as pkl
data = np.load('/data2/lrh/project/dance/double/DoubleExp/experiments/Local_Module/Norm_DoubleDance_655/val1999/samples_2024-03-25-05-28-02/000_1999_a.npy')
print(data.shape)



# data2 = np.load('/data2/lrh/dataset/fine_dance/RepairedDouble/after_split/vertices/30fps/new_xyz_vecs_30fps/001_0.npy')
# print(data2.shape)

# with open('/data2/lrh/project/dance/long/experiments/1004_edge139_128/infer/2730/test_inp_0_036z@017.pkl','rb') as f:
#     data2 = pkl.load(f)
# print(data2.shape)