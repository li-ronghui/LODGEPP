import pickle
import os
import numpy as np

def get_key_list(musicpeak_file, motionbeat_file, origin_modir, load_mobeat=False, only_beat=False):
    key_idx = {}
    with open(musicpeak_file, "rb") as f1:
        music_peak = pickle.load(f1)
    with open(motionbeat_file, "rb") as f2:
        motion_beat = pickle.load(f2)
    
    for key_ in motion_beat.keys():
        key = key_[1:] if key_[0] == 'M' else key_
        if only_beat:
            temp_list = list(set(music_peak[key]['beat_idxs'].tolist()))
        else:
            if load_mobeat:
                temp_list = list(set(music_peak[key]['beat_idxs'].tolist()      \
                                        + music_peak[key]['onset_idxs'].tolist()    \
                                        + motion_beat[key_].tolist()                \
                                                ))
            else:
                temp_list = list(set(music_peak[key]['beat_idxs'].tolist()      \
                                        + music_peak[key]['onset_idxs'].tolist()    \
                                                ))
                                            
        motion = np.load(os.path.join(origin_modir, key_+'.npy'))
        length = motion.shape[0]
        child_list = [x for x in temp_list if x < length]
        child_list.sort()
        key_idx[key_] = child_list

    return key_idx

if __name__ == "main":
    print("a")
    origin_modir = "dataset/data/origin/mofeature_vq_full"
    key_list = get_key_list("tools/musicpeak.pkl","tools/motionbeat.pkl","dataset/data/origin/mofeature_vq_full")

    print(key_list)
