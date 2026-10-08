import argparse
import yaml
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from omegaconf import OmegaConf

def parse_args_diffusion(phase="train", condition=False):
    # Use ArgumentDefaultsHelpFormatter to include default values in help
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # Common arguments
    parser.add_argument("--feature_type", type=str, default="baseline")
    parser.add_argument(
        "--full_seq_len", type=int, default=2048, help="full_seq_len"
    ) 
    parser.add_argument("--datasplit", type=str, default="cross_dancer", choices=["cross_genre", "cross_dancer"])
    parser.add_argument(
        "--windows", type=int, default=300, help="windows"
    ) 
    parser.add_argument(
        "--cond_feadim", type=int, default=35, help="cond_feadim"
    )
    parser.add_argument(
        "--fd-motion-dir", type=str, default='/data/lrh/datasets/fine_dance/origin/motion_feature319mirror', help="music_dir_list"
    )
    parser.add_argument("--out_length", type=float, default=30, help="max. length of output, in seconds")
    parser.add_argument(
        "--render_dir", type=str, default="FineDance_test_renders/", help="Sample render path"
    )
    parser.add_argument(
        "--mudic_feature_dir", type=str, default="/data2/lrh/dataset/fine_dance/div_by_time/music_25", help="mudic_feature_dir"
    )
    parser.add_argument(
        "--keymotion_dir", type=str, default='', help="keymotion_dir"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoint.pt", help="checkpoint"
    )
    parser.add_argument(
        "--nfeats", type=int, default=139, help="nfeats"
    )
    parser.add_argument(
        "--music_dir",
        type=str,
        default="data/test/wavs",
        help="folder containing input music",
    )
    parser.add_argument(
        "--save_motions", action="store_true", help="Saves the motions for evaluation"
    )
    parser.add_argument(
        "--fullpt", action="store_true", help="Saves the full model"
    )
    parser.add_argument(
        "--motion_save_dir",
        type=str,
        default="eval/motions",
        help="Where to save the motions",
    )
    parser.add_argument(
        "--cache_features",
        action="store_true",
        help="Save the jukebox features for later reuse",
    )
    parser.add_argument(
        "--do_normalize",
        action="store_true",
        help="normalize",
    )
    parser.add_argument(
        "--no_render",
        action="store_true",
        help="Don't render the video",
    )
    parser.add_argument(
        "--use_cached_features",
        action="store_true",
        help="Use precomputed features instead of music folder",
    )
    parser.add_argument(
        "--feature_cache_dir",
        type=str,
        default="cached_features/",
        help="Where to save/load the features",
    )
    parser.add_argument(
        "--keystage",
        action="store_true",
        help="wheather in the key generation stage",
    )
    parser.add_argument(
        "--no_vloss",
        action="store_true",
        help="wheather use_vloss",
    )
    parser.add_argument(
        "--decoder_type", type=str, default="hrnet", help="decoder_type"
    )
    parser.add_argument(
        "--scale", type=int, default=8, help="scale"
    ) 
    parser.add_argument(
        "--hint", type=int, default=0, help="hint"
    ) 
    parser.add_argument(
        "--hryaml", type=str, default="configs/hryaml.yaml", help="hryaml"
    )
    parser.add_argument("--gpu", type=int, default=5)
    parser.add_argument("--mode", type=str, default="inpaint_soft")
    parser.add_argument("--soft_hint", type=str, default="gpt")
    parser.add_argument("--length1", type=int, default=512)
    parser.add_argument("--length2", type=int, default=128)
    parser.add_argument("--gpt_window_size", type=int, default=128)
    parser.add_argument("--clip_frames", type=int, default=4)
    # parser.add_argument("--checkpoint1", type=str, default="experiments/edge139_Coarse1024/train/mu2565/weights/Full-train-3320.pt")
    # parser.add_argument("--checkpoint2", type=str, default="experiments/edge139_256/1005edge139_256_35/train/weights/train-2500.pt")
    # parser.add_argument('--checkpoint2', type=str, default='experiments/1023edge139_256_35/train/bce_fc2/weights/Full-train-3070.pt')
    parser.add_argument("--nb-code", type=int, default=1024, help="nb of embedding") # 数据集扩大, 增大一倍
    parser.add_argument("--code-dim", type=int, default=1024, help="embedding dimension")
    parser.add_argument("--output-emb-width", type=int, default=1024, help="output embedding width")
    parser.add_argument("--tk_dir", type=str, default="/data/lrh/project/dance/LongV2/LongV2Exp1/experiments/vqvae/output_vq/mo266/smplxmo266_0401_win64_Norm/eval_idx_flat")
    # good
    parser.add_argument("--resume-trans", type=str, default="/data/lrh/project/dance/LongV2/LongV2Exp1/experiments/vqvae/gpt/mo266/smplxmo266_0401_win64_Norm/ckpt/mask_best_acc.pth")
    parser.add_argument("--resume-pth", type=str, default="/data/lrh/project/dance/LongV2/LongV2Exp1/experiments/vqvae/output_vq/mo266/smplxmo266_0401_win64_Norm/vqvae5870/f_266/best_recon.pth")
    # compare
    # parser.add_argument("--resume-trans", type=str, default="experiments/vqvae/gpt/clip8_139/FineDance_1009_gpt_win128/ckpt/mask_last.pth")
    # parser.add_argument("--resume-pth", type=str, default="experiments/vqvae/output_vq/clip8_139/FineDance_1011_2048_win128/vqvae/f_139/best_conmit.pth")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=1024, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--vq-norm', type=str, default=None, help='dataset directory')
    parser.add_argument("--block-size", type=int, default=2049)
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument("--embed-dim-gpt", type=int, default=512, help="embedding dimension")
    parser.add_argument("--clip-dim", type=int, default=512, help="latent dimension in the clip feature")
    parser.add_argument("--num-layers", type=int, default=2, help="nb of transformer layers")
    parser.add_argument("--n-head-gpt", type=int, default=8, help="nb of heads")
    parser.add_argument("--ff-rate", type=int, default=4, help="feedforward size")
    parser.add_argument("--drop-out-rate", type=float, default=0.1, help="dropout ratio in the pos encoding")

    # Phase-specific arguments
    group = parser.add_argument_group(f"{phase.capitalize()} options")
    if phase in ["train", "test", "demo"]:
        group.add_argument(
            "--cfg",
            type=str,
            required=False,
            default="/data/jay/project/longdance-ground/DanceDiffusion/configs/lodge/45/finedance_fea338.yaml",
            help="config file",
        )
        group.add_argument(
            "--cfg_assets",
            type=str,
            required=False,
            default="/data/jay/project/longdance-ground/DanceDiffusion/configs/40assets.yaml",
            help="config file for asset paths",
        )
        group.add_argument("--batch_size",
                           type=int,
                           required=False,
                           help="training batch size")
        group.add_argument("--device",
                           type=int,
                           nargs="+",
                           required=False,
                           help="training device")
        group.add_argument("--nodebug",
                           action="store_true",
                           required=False,
                           help="debug or not")
        group.add_argument("--dir",
                           type=str,
                           required=False,
                           help="evaluate existing npys")

    if phase == "demo":
        # group.add_argument("--motion_transfer", action='store_true', help="Motion Distribution Transfer")
        group.add_argument("--render",
                           action="store_true",
                           help="Render visulizaed figures")
        group.add_argument("--render_mode", type=str, help="video or sequence")
        group.add_argument(
            "--frame_rate",
            type=float,
            default=30,
            help="the frame rate for the input/output motion",
        )
        group.add_argument(
            "--replication",
            type=int,
            default=1,
            help="the frame rate for the input/output motion",
        )
        group.add_argument(
            "--example",
            type=str,
            required=False,
            help="input text and lengths with txt format",
        )
        group.add_argument(
            "--task",
            type=str,
            required=False,
            help="random_sampling, reconstrucion or text_motion",
        )
        group.add_argument(
            "--out_dir",
            type=str,
            required=False,
            help="output dir",
        )
        group.add_argument(
            "--allinone",
            action="store_true",
            required=False,
            help="output seperate or combined npy file",
        )

    if phase == "render":
        group.add_argument(
            "--cfg",
            type=str,
            required=False,
            default="./configs/render.yaml",
            help="config file",
        )
        group.add_argument(
            "--cfg_assets",
            type=str,
            required=False,
            default="./configs/assets.yaml",
            help="config file for asset paths",
        )
        # group.add_argument("--motion_transfer", action='store_true', help="Motion Distribution Transfer")
        group.add_argument("--npy",
                           type=str,
                           required=False,
                           default=None,
                           help="npy motion files")
        group.add_argument("--dir",
                           type=str,
                           required=False,
                           default=None,
                           help="npy motion folder")
        group.add_argument(
            "--mode",
            type=str,
            required=False,
            default="sequence",
            help="render target: video, sequence, frame",
        )
        group.add_argument(
            "--joint_type",
            type=str,
            required=False,
            default=None,
            help="mmm or vertices for skeleton",
        )
    # update config from files
    params = parser.parse_args()
    cfg_exp = OmegaConf.load(params.cfg)
    cfg_assets = OmegaConf.load(params.cfg_assets)
    cfg = OmegaConf.merge(cfg_exp, cfg_assets)

    # These parameters can set by '--'
    if phase in ["train", "test"]:
        cfg.TRAIN.BATCH_SIZE = (params.batch_size
                                if params.batch_size else cfg.TRAIN.BATCH_SIZE)
        cfg.DEVICE = params.device if params.device else cfg.DEVICE
        cfg.DEBUG = not params.nodebug if params.nodebug is not None else cfg.DEBUG

        # no debug in test
        cfg.DEBUG = False if phase == "test" else cfg.DEBUG
        if phase == "test":
            cfg.DEBUG = False
            # cfg.DEVICE = [0]
            print("Force no debugging and one gpu when testing")
        cfg.TEST.TEST_DIR = params.dir if params.dir else cfg.TEST.TEST_DIR

    if phase == "demo":
        # cfg.DEMO.MOTION_TRANSFER = params.motion_transfer
        cfg.DEVICE = params.device if params.device else cfg.DEVICE
        cfg.DEMO.RENDER = params.render
        cfg.DEMO.FRAME_RATE = params.frame_rate
        cfg.DEMO.EXAMPLE = params.example
        cfg.DEMO.TASK = params.task
        cfg.TEST.FOLDER = params.out_dir if params.dir else cfg.TEST.FOLDER
        cfg.DEMO.REPLICATION = params.replication
        cfg.DEMO.OUTALL = params.allinone

    if phase == "render":
        if params.npy:
            cfg.RENDER.NPY = params.npy
            cfg.RENDER.INPUT_MODE = "npy"
        if params.dir:
            cfg.RENDER.DIR = params.dir
            cfg.RENDER.INPUT_MODE = "dir"
        cfg.RENDER.JOINT_TYPE = params.joint_type
        cfg.RENDER.MODE = params.mode

    # debug mode
    if cfg.DEBUG:
        cfg.NAME = "debug--" + cfg.NAME
        cfg.LOGGER.WANDB.OFFLINE = True
        # cfg.LOGGER.VAL_EVERY_STEPS = 1
    if condition:
        return parser, cfg
    else:
        return params, cfg