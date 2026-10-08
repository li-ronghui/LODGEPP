import torch, sys
sys.path.append('/data2/lrh/project/dance/LodgePlus/Lodge_plus_smpl/DanceDiffusion')


ckpt = torch.load('/data2/lrh/project/dance/Lodge/lodge302/data/Normalizer.pth')
print(ckpt)