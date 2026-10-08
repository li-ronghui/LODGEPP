import copy
import glob
import torch
from torch.utils import data
import numpy as np
import os
from tqdm import tqdm
from os.path import join as opj
import json
# import torchgeometry as tgy
# from dld.data.utils.smplfk import set_on_ground_139, SMPLX_Skeleton
import sys


Song2Genre = {
    "001": "Jazz",
    "002": "Chinese",
    "003": "Urban",
    "004": "Jazz",
    "005": "Choreography",
    "006": "ShenYun",
    "007": "ShenYun",
    "008": "ShenYun",
    "009": "ShenYun",
    "010": "ShenYun",
    "011": "ShenYun",
    "012": "HanTang",
    "013": "HanTang",
    "014": "ShenYun",
    "015": "ShenYun",
    "016": "Miao",
    "017": "Miao",
    "018": "Wei",
    "019": "Dai",
    "020": "Dai",
    "021": "Wei",
    "022": "Wei",
    "023": "Waltz",
    "024": "Waltz",
    "025": "Waltz",
    "026": "Tango",
    "027": "Chacha",
    "028": "Jive",
    "029": "Rumba",
    "030": "Rumba",
    "031": "Rumba",
    "032": "Samba",
    "033": "Rumba",
    "034": "Chacha",
    "035": "Samba",
    "036": "Rumba",
    "037": "Samba",
    "038": "Rumba",
    "039": "Rumba",
    "040": "Chacha",
    "041": "Chacha",
    "042": "Samba",
    "043": "Jive",
    "044": "Jive",
    "045": "Kpop",
    "046": "Kpop",
    "047": "Kpop",
    "048": "Kpop",
    "049": "Kpop",
    "050": "Kpop",
    "051": "Kpop",
    "052": "Kpop",
    "053": "Kpop",
    "054": "Kpop",
    "055": "Jazz",
    "056": "Urban",
    "057": "Urban",
    "058": "Urban",
    "059": "Kpop",
    "060": "Kpop",
    "061": "Kpop",
    "062": "Kpop",
    "063": "Kpop",
    "064": "Kpop",
    "065": "Kpop",
    "066": "Kpop",
    "067": "HipHop",
    "068": "HipHop",
    "069": "HipHop",
    "070": "HipHop",
    "071": "HipHop",
    "072": "HipHop",
    "073": "HipHop",
    "074": "HipHop",
    "075": "HipHop",
    "076": "HipHop",
    "077": "DunHuang",
    "078": "Dai",
    "079": "DunHuang",
    "080": "HanTang",
    "081": "HanTang",
    "082": "HanTang",
    "083": "Dai",
    "084": "DunHuang",
    "085": "Wei",
    "086": "Wei",
    "087": "Dai",
    "088": "Dai",
    "089": "Wei",
    "090": "Waltz",
    "091": "Waltz",
    "092": "Waltz",
    "093": "Tango",
    "094": "Samba",
    "095": "Samba",
    "096": "Kpop",
    "097": "Kpop",
    "098": "Kpop",
    "099": "HanTang"
}
DanceGenre = {'Jazz': '0', 'Chinese': '1', 'Urban': '2', 'Choreography': '3', 'ShenYun': '4', 'HanTang': '5', 'Miao': '6', 'Wei': '7', 'Dai': '8', 'Waltz': '9', 'Tango': '10', 'Chacha': '11', 'Jive': '12', 'Rumba': '13', 'Samba': '14', 'Kpop': '15', 'HipHop': '16', 'DunHuang': '17'}


def is_odd_multiple(num, base):
    if num % base != 0:
        return False
    multiple = num // base
    return multiple % 2 != 0

class DoubleDance_split(data.Dataset):
    def __init__(self, args, istrain, dataname=None):
        self.motion_dir =  eval(f"args.DATASET.{dataname.upper()}.MOTION")     #'/data/lrh/datasets/fine_dance/origin/motion_feature319'
        self.music_dir = eval(f"args.DATASET.{dataname.upper()}.MUSIC")        #'/data/lrh/datasets/fine_dance/origin/music_feature35'
        self.song2genre = Song2Genre

        # sys.exit(0)
        leader = args.DATA_SETTING.leader
        self.istrain = istrain
        self.args = args
        test_list = ['008','062','072','033','057']

        self.leader = []
        self.follower = []
        self.music = []
        self.name = []
        for song in tqdm(range(1, 100)):          # !debug 100
            song_name = str(song).zfill(3)
            if self.istrain:
                if song_name in test_list:
                    continue
            else:
                if not song_name in test_list:
                    continue
            
            if args.DATA_SETTING.mirror:
                molist_leader = glob.glob(opj(self.motion_dir, '*' + song_name + '_0@*' + leader + '.npy' ))
                molist_follower = glob.glob(opj(self.motion_dir, '*' + song_name + '_1@*' + leader +  '.npy'))
            else:
                molist_leader = glob.glob(opj(self.motion_dir, song_name + '_0@*' + leader +  '.npy'))
                molist_follower = glob.glob(opj(self.motion_dir, song_name + '_1@*' + leader +  '.npy'))

            molist_leader.sort()
            self.leader = self.leader + molist_leader
            molist_follower.sort()
            self.follower = self.follower + molist_follower

        for file in copy.deepcopy(self.follower):
            basename = os.path.basename(file).split('.')[0]
            index = basename.split('@')[1]

            # 是否减少一部分数据
            # if is_odd_multiple(int(index), 32):
            #     self.follower.remove(file)
            #     leader_file = opj(self.motion_dir, basename.split('_')[0] + '_0@'  + str(index) + '@' + leader +  '.npy')
            #     self.leader.remove(leader_file)
            #     continue

            music_name = basename.split('_')[0] + '@' + basename.split('@')[1] + '.npy'
            mufile = opj(self.music_dir, music_name)
            if os.path.exists(mufile):
                self.music.append(mufile)
                self.name.append(basename)
            else:
                continue

        self.len = len(self.name)
        print(f'Dataset has {self.len} samples..')

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        leader = np.load(self.leader[index])
        follower = np.load(self.follower[index])
        music = np.load(self.music[index])
        name = self.name[index]

        return leader, follower, music, name
    
    def get_train_test_list(self):
        all_list = []
        train_list = []
        for i in range(1,100):
            all_list.append(str(i).zfill(3))
  
        test_list = ['008','062','072','033','057']
     
        ignor_list = []
        for one in all_list:
            if one not in test_list:
                train_list.append(one)
                
        return ignor_list, train_list, test_list
    
if __name__ == '__main__':
    # Genre_double = {}
    # genre_li = []
    # idx = 0
    # for one in Song2Genre.values():
    #     print(one)
    #     if not one in genre_li:
    #         genre_li.append(one)
    
    # idx = 0
    # for ge in genre_li:
    #     Genre_double[ge] = str(idx)
    #     idx += 1
    # print(genre_li)
    # print(Genre_double)

    print(is_odd_multiple(32,32))
    print(is_odd_multiple(64,32))
    print(is_odd_multiple(96,32))
    print(is_odd_multiple(128,32))
    print(is_odd_multiple(32,32))