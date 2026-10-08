import argparse
import yaml

def FineDance_parse_train_opt(condition=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="FineDance_runs/train", help="project/name")
    parser.add_argument("--exp_name", default="FineDance_exp_0601_", help="save to project/name")
    parser.add_argument("--feature_type", type=str, default="baseline")
    # parser.add_argument("--data_path", type=str, default="/data2/lrh/dataset/fine_dance/origin/motion_feature319", help="raw data path")
    parser.add_argument("--datasplit", type=str, default="cross_dancer", choices=["cross_genre", "cross_dancer"])
    parser.add_argument(
        "--render_dir", type=str, default="FineDance_renders/", help="Sample render path"
    )
    parser.add_argument(
        "--full_seq_len", type=int, default=150, help="full_seq_len"
    ) 
    parser.add_argument(
        "--cond_feadim", type=int, default=35+135, help="cond_feadim"
    )
    parser.add_argument(
        "--fd-motion-dir", type=str, default='/data/lrh/datasets/fine_dance/origin/motion_feature319mirror', help="music_dir_list"
    )
    parser.add_argument(
        "--music_dir_list", type=str, default='/data2/lrh/dataset/fine_dance/div_by_time/music_25', help="music_dir_list"
    )
    parser.add_argument(
        "--windows", type=int, default=10, help="windows"
    ) 
    parser.add_argument(
        "--mix", action="store_true", help="Saves the motions for evaluation"
    )
    # parser.add_argument("--feature_type", type=str, default="jukebox")
    parser.add_argument(
        "--wandb_pj_name", type=str, default="FineDance_wandb", help="project name"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")        # default=64
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument(
        "--save_interval",
        type=int,
        default=10,            # default=100,  
        help='Log model after every "save_period" epoch',
    )
    parser.add_argument("--ema_interval", type=int, default=1, help="ema every x steps")
    parser.add_argument(
        "--checkpoint", type=str, default="", help="trained checkpoint path (optional)"
    )
    parser.add_argument(
        "--do_normalize",
        action="store_true",
        help="normalize",
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
        "--nfeats", type=int, default=139, help="nfeats"
    ) 
    parser.add_argument(
        "--decoder_type", type=str, default="hrnet", help="decoder_type"
    ) 
    parser.add_argument(
        '--decoder_dim', 
        nargs='+', 
        type=int, 
        help='List of numbers',
        default=[1024, 768, 512, 139],
    )
    parser.add_argument(
        "--scale", type=int, default=8, help="scale"
    ) 
    parser.add_argument(
        "--keymotion_dir", type=str, default='', help="keymotion_dir"
    )
    parser.add_argument(
        "--hryaml", type=str, default="configs/EDGE_KeyDiff_Hrnet/hryaml.yaml", help="hryaml"
    )
    parser.add_argument(
        "--baseyaml", type=str, default="configs/EDGE_KeyDiff_Hrnet/base.yaml", help="hryaml"
    )
    parser.add_argument(
        "--hint", type=int, default=0, help="hint"
    ) 
    parser.add_argument(
        "--fullpt", action="store_true", help="Saves the motions for evaluation"
    )
    parser.add_argument("--partial", type=str, default="full", choices=["full", "morden", "tradition"])
    if condition:
        return parser
    else:
        opt = parser.parse_args()
        return opt

def FineDance_parse_test_opt(condition=False): 
    parser = argparse.ArgumentParser()
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
    
    if condition:
        return parser
    else:
        opt = parser.parse_args()
        return opt


def save_arguments_to_yaml(args, file_path):
    arg_dict = vars(args)  # 将Namespace对象转换为字典
    yaml_str = yaml.dump(arg_dict, default_flow_style=False)

    with open(file_path, 'w') as file:
        file.write(yaml_str)