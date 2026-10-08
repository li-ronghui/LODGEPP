from fileinput import filename
import glob
import os
import pickle
import numpy as np

import librosa as lr
import numpy as np
import soundfile as sf
from tqdm import tqdm


def slice_audio(audio_file, stride, length, out_dir):
    # stride, length in seconds
    audio, sr = lr.load(audio_file, sr=None)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    start_idx = 0
    idx = 0
    window = int(length * sr)
    stride_step = int(stride * sr)
    while start_idx <= len(audio) - window:
        audio_slice = audio[start_idx : start_idx + window]
        sf.write(f"{out_dir}/{file_name}_slice{idx}.wav", audio_slice, sr)
        start_idx += stride_step
        idx += 1
    return idx


def slice_motion(motion_file, stride, length, num_slices, out_dir):
    motion = pickle.load(open(motion_file, "rb"))
    pos, q = motion["pos"], motion["q"]
    scale = motion["scale"][0]

    file_name = os.path.splitext(os.path.basename(motion_file))[0]
    # normalize root position
    pos /= scale
    start_idx = 0
    window = int(length * 60)
    stride_step = int(stride * 60)
    slice_count = 0
    # slice until done or until matching audio slices
    while start_idx <= len(pos) - window and slice_count < num_slices:
        pos_slice, q_slice = (
            pos[start_idx : start_idx + window],
            q[start_idx : start_idx + window],
        )
        out = {"pos": pos_slice, "q": q_slice}
        pickle.dump(out, open(f"{out_dir}/{file_name}_slice{slice_count}.pkl", "wb"))
        start_idx += stride_step
        slice_count += 1
    return slice_count

def slice_magic_motion(motion_file, stride, length, num_slices, out_dir):
    motion = np.load(motion_file)
    pos = motion[:,:3]
    q = motion[:, 3:]

    file_name = motion_file.split("/")[-1].split(".")[0] #os.path.splitext(os.path.basename(motion_file))[0]
    # normalize root position
    # scale = 1
    # pos /= scale
    start_idx = 0
    window = int(length * 30)
    stride_step = int(stride * 30)
    slice_count = 0
    # slice until done or until matching audio slices
    while start_idx <= len(pos) - window and slice_count < num_slices:
        pos_slice, q_slice = (
            pos[start_idx : start_idx + window],
            q[start_idx : start_idx + window],
        )
        out = {"pos": pos_slice, "q": q_slice}
        pickle.dump(out, open(f"{out_dir}/{file_name}_slice{slice_count}.pkl", "wb"))
        start_idx += stride_step
        slice_count += 1
    return slice_count


def slice_aistpp(motion_dir, wav_dir, stride=0.5, length=5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    motions = sorted(glob.glob(f"{motion_dir}/*.pkl"))
    wav_out = wav_dir + "_sliced"
    motion_out = motion_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    os.makedirs(motion_out, exist_ok=True)
    assert len(wavs) == len(motions)
    for wav, motion in tqdm(zip(wavs, motions)):
        # make sure name is matching
        m_name = os.path.splitext(os.path.basename(motion))[0]
        w_name = os.path.splitext(os.path.basename(wav))[0]
        assert m_name == w_name, str((motion, wav))
        audio_slices = slice_audio(wav, stride, length, wav_out)
        motion_slices = slice_motion(motion, stride, length, audio_slices, motion_out)
        # make sure the slices line up
        assert audio_slices == motion_slices, str(
            (wav, motion, audio_slices, motion_slices)
        )

def slice_magicsmpl(opt, list_, ign_list, outdir, stride=0.5, length=5):
    wav_out_dir = os.path.join(outdir, 'wavs')
    dance_out_dir = os.path.join(outdir, 'dances')
    os.makedirs(wav_out_dir, exist_ok=True)
    os.makedirs(dance_out_dir, exist_ok=True)
    for one in tqdm(list_):
        if one not in ign_list:
            audiofile = os.path.join(opt.music_path, one+".wav")
            dancefile = os.path.join(opt.dance_path, one+'.npy')
            
            audio_slices = slice_audio(audiofile, stride, length, wav_out_dir)
            motion_slices = slice_magic_motion(dancefile, stride, length, audio_slices, dance_out_dir)
            # make sure the slices line up
            # assert audio_slices == motion_slices, str(
            #     (audiofile, dancefile, audio_slices, motion_slices)
            # )
    

def slice_audio_folder(wav_dir, stride=0.5, length=5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    wav_out = wav_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    for wav in tqdm(wavs):
        audio_slices = slice_audio(wav, stride, length, wav_out)
