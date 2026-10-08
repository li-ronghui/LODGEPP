import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # dataloader-mixed data
    parser.add_argument('--window_size', type=int, default=64, help='training motion length')
    parser.add_argument('--gpu', type=int, default=3, help='training motion length')
    parser.add_argument('--feature_dim', type=int, default=135,choices=[135,263,139], help='only render feature or with additional feature')
    parser.add_argument('--tk_dir',default='/opt/data/private/control2dance/debug_exp/dataset/data/tokenized/f_135/',type = str,help='Tokenized humanml3d+aistpp+finedance data root dir')
    parser.add_argument('--batch-size', default=2, type=int, help='debug batch size') # debug
    # parser.add_argument('--batch-size', default=128, type=int, help='batch size') # origin
    # parser.add_argument('--batch-size', default=256, type=int, help='batch size')
    parser.add_argument('--mask-len', default=25, type=int, help='masked seq len') # debug

    # t2m-gpt
    # parser.add_argument('--fps', default=[20], nargs="+", type=int, help='frames per second')
    # parser.add_argument('--seq-len', type=int, default=64, help='training motion length')

    # dataloader-finedance
    parser.add_argument('--full-seq-len', default = 128, type=int, help='frame rate of a dance sequence')
    parser.add_argument('--mix', default = False, type=bool, help='number of total iterations to run')
    parser.add_argument('--fd-music-dir', default='/opt/data/private/control2dance/debug_exp/dataset/data/origin/music_feature35_edge', type=str, help='music dir')
    parser.add_argument('--fd-motion-dir', default='/opt/data/private/control2dance/debug_exp/dataset/data/mofeatrue135/mofeature_fd', type=str, help='motion dir')
    parser.add_argument('--fd-g-dir', default='/opt/data/private/control2dance/debug_exp/dataset/data/origin/label_json/', type=str, help='genre lable dir')
    

    # dataloader-aist++ 
    parser.add_argument('--aistpp_dir', default = '/opt/data/private/control2dance/debug_exp/dataset/data/aistpp/aist_plusplus_final/', type=str, help='Aist++ dir')
    parser.add_argument('--aistpp_p_dir', default = '/opt/data/private/control2dance/debug_exp/dataset/data/mofeatrue135/mofeature_aistpp/', type=str, help='Processed Aist++ dir')
    parser.add_argument('--aistpp_p_music_dir', default = '/opt/data/private/control2dance/debug_exp/dataset/data/aistpp/music_feature35/', type=str, help='Aist++ processed music dir')
    parser.add_argument('--aistpp_smpl_dir', default = '/opt/data/private/control2dance/debug_exp/dataset/data/aistpp/smpl/', type=str, help='Aist++ SMPL(MALE) dir')

    # dataloader-HumanML3D
    parser.add_argument('--p_h3d_dir', default = '/opt/data/private/control2dance/debug_exp/dataset/data/mofeatrue135/mofeature_h3d/', type=str, help='Processed amass(humanml3d) data root dir')
    parser.add_argument('--h3d_texts_dir', default = '/opt/data/private/control2dance/debug_exp/dataset/data/HumanML3D/texts/', type=str, help='HumanML3D data index.csv')
    parser.add_argument('--o_h3d_dir',default='/opt/data/private/control2dance/debug_exp/dataset/data/HumanML3D/',type = str,help='Original amass(humanml3d) data root dir')
    

    ## optimization
    # parser.add_argument('--total-iter', default=100000, type=int, help='number of total iterations to run') # origin
    parser.add_argument('--total-iter', default=200000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[60000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")
    
    parser.add_argument('--weight-decay', default=1e-6, type=float, help='weight decay') 
    parser.add_argument('--decay-option',default='all', type=str, choices=['all', 'noVQ'], help='disable weight decay on codebook')
    parser.add_argument('--optimizer',default='adamw', type=str, choices=['adam', 'adamw'], help='disable weight decay on codebook')
    
    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=512, help="embedding dimension")
    # parser.add_argument("--nb-code", type=int, default=512, help="nb of embedding")
    parser.add_argument("--nb-code", type=int, default=1024, help="nb of embedding") # 数据集扩大, 增大一倍
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--vq-norm', type=str, default=None, help='dataset directory')
    
    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument('--beta', type=float, default=1.0, help='commitment loss in standard VQ')

    ## gpt arch
    # parser.add_argument("--block-size", type=int, default=25, help="seq len")
    parser.add_argument("--block-size", type=int, default=121, help="seq len")
    parser.add_argument("--embed-dim-gpt", type=int, default=512, help="embedding dimension")
    parser.add_argument("--clip-dim", type=int, default=512, help="latent dimension in the clip feature")
    parser.add_argument("--num-layers", type=int, default=2, help="nb of transformer layers")
    parser.add_argument("--n-head-gpt", type=int, default=8, help="nb of heads")
    parser.add_argument("--ff-rate", type=int, default=4, help="feedforward size")
    parser.add_argument("--drop-out-rate", type=float, default=0.1, help="dropout ratio in the pos encoding")

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume pth for VQ')
    # parser.add_argument("--resume-pth", type=str, default='/opt/data/private/control2dance/debug_exp/ckpt/vqvae/f_135/last.pth', help='resume pth for VQ')
    parser.add_argument("--resume-trans", type=str, default=None, help='resume pth for GPT')
    # parser.add_argument("--resume-trans", type=str, default='/opt/data/private/control2dance/debug_exp/ckpt/gpt/f_135/last.pth', help='resume pth for GPT')
    
    
    
    
    ## output directory 
    parser.add_argument('--out-dir', type=str, default='output_gpt/', help='output directory')
    parser.add_argument('--ckpt-dir', type=str, default='/opt/data/private/control2dance/debug_exp/ckpt/', help='ckpt directory')
    parser.add_argument('--results-dir', type=str, default='visual_results/', help='output directory')
    parser.add_argument('--visual-name', type=str, default='baseline', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out-dir')
    ## other
    parser.add_argument('--print-iter', default=10, type=int, help='print frequency') # debug
    # parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--save-iter', default=50, type=int, help='save frequency')
    parser.add_argument('--eval-iter', default=1000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')
    
    parser.add_argument('--vis-gt', action='store_true', help='whether visualize GT motions')
    parser.add_argument('--nb-vis', default=20, type=int, help='nb of visualizations')
    
    parser.add_argument("--if-maxtest", action='store_true', help="test in max")
    parser.add_argument('--pkeep', type=float, default=1.0, help='keep rate for gpt training')
    
    return parser.parse_args()