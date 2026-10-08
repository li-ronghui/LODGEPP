import argparse

def get_args_parser():
    parser = argparse.ArgumentParser(description='Optimal Transport AutoEncoder training for AIST',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # mixed data
    parser.add_argument('--dataset', type=str, nargs='+', default=["all"], help='training dataset')     
    parser.add_argument('--normdir', type=str, default='/data/lrh/datasets/fine_dance/gound/smplx_mofea266/noMirror_Norm', help='training dataset')
    parser.add_argument('--gpu', type=int, default=1, help='gpu')
    parser.add_argument('--window_size', type=int, default=64, help='training motion length')
    parser.add_argument('--windows', type=int, default=1, help='number of windows')
    parser.add_argument('--feature_dim', type=int, default=135,choices=[135,139,266,263], help='only render feature or with additional feature')
    parser.add_argument('--tk_dir',default='/data2/lrh/dyq/debug_exp/dataset/data/tokenized/f_135/',type = str,help='Tokenized humanml3d+aistpp+finedance data root dir')

    # dataloader-finedance
    parser.add_argument('--full-seq-len', default = 128, type=int, help='frame rate of a dance sequence')
    parser.add_argument('--mix', default = False, type=bool, help='number of total iterations to run')
    parser.add_argument('--batch-size', default=128, type=int, help='batch size')
    # parser.add_argument('--batch-size', default=256, type=int, help='batch size')
    parser.add_argument('--fd-music-dir', default='/data2/lrh/dyq/debug_exp/dataset/data/origin/music_feature35_edge', type=str, help='music dir')
    parser.add_argument('--fd-motion-dir', default='/data2/lrh/dyq/debug_exp/dataset/data/origin/mofeature_vq', type=str, help='motion dir')
    parser.add_argument('--partial', default='full', type=str, help='weather use partial data')

    # dataloader-aist++
    parser.add_argument('--aistpp_dir', default = '/data2/lrh/human_datasets/aist_plusplus_final/', type=str, help='Aist++ dir')
    parser.add_argument('--aistpp_p_dir', default = '/data2/lrh/dyq/debug_exp/dataset/data/mofeatrue135/mofeature_aistpp/', type=str, help='Processed Aist++ dir')

    # dataloader-HumanML3D
    parser.add_argument('--p_h3d_dir', default = '/data2/lrh/dyq/debug_exp/dataset/data/mofeatrue135/', type=str, help='Processed amass(humanml3d) data root dir')
    parser.add_argument('--h3d_texts_dir', default = '/data2/lrh/dyq/debug_exp/dataset/data/HumanML3D/texts/', type=str, help='HumanML3D data index.csv')
    parser.add_argument('--o_h3d_dir',default='/data2/lrh/dyq/debug_exp/dataset/data/HumanML3D/',type = str,help='Original amass(humanml3d) data root dir')


    ## optimization
    parser.add_argument('--total-iter', default=200000, type=int, help='number of total iterations to run')
    parser.add_argument('--warm-up-iter', default=1000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--lr-scheduler', default=[50000, 400000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.05, type=float, help="learning rate decay")

    parser.add_argument('--weight-decay', default=0.0, type=float, help='weight decay')
    parser.add_argument("--commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")
    parser.add_argument('--loss-vel', type=float, default=1, help='hyper-parameter for the velocity loss')
    parser.add_argument('--recons-loss', type=str, default='l2', help='reconstruction loss')
    
    ## vqvae arch
    parser.add_argument("--code-dim", type=int, default=128, help="embedding dimension")
    parser.add_argument("--nb-code", type=int, default=512, help="nb of embedding")
    # parser.add_argument("--nb-code", type=int, default=8192, help="nb of embedding") # 数据集扩大, 增大一倍
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down-t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride-t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=128, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="depth of the network")
    parser.add_argument("--dilation-growth-rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output-emb-width", type=int, default=128, help="output embedding width")
    parser.add_argument('--vq-act', type=str, default='relu', choices = ['relu', 'silu', 'gelu'], help='dataset directory')
    parser.add_argument('--vq-norm', type=str, default=None, help='dataset directory')
    
    ## quantizer
    parser.add_argument("--quantizer", type=str, default='ema_reset', choices = ['ema', 'orig', 'ema_reset', 'reset'], help="eps for optimal transport")
    parser.add_argument('--beta', type=float, default=1.0, help='commitment loss in standard VQ')

    ## resume
    parser.add_argument("--resume-pth", type=str, default=None, help='resume pth for VQ')
    # parser.add_argument("--resume-pth", type=str, default='/data2/lrh/dyq/debug_exp/ckpt/vqvae/f_135/last.pth', help='resume pth for VQ')  # default='/opt/data/private/control2dance/debug_exp/ckpt/vqvae/f_135/last.pth
    parser.add_argument("--resume-gpt", type=str, default=None, help='resume pth for GPT')
    
    
    ## output directory 
    parser.add_argument('--out-dir', type=str, default='experiments/vqvae/output_vq/', help='output directory')
    parser.add_argument('--ckpt-dir', type=str, default='experiments/vqvae/ckpt', help='ckpt directory')
    parser.add_argument('--results-dir', type=str, default='visual_results/', help='output directory')
    parser.add_argument('--visual-name', type=str, default='baseline', help='output directory')
    parser.add_argument('--exp-name', type=str, default='exp_debug', help='name of the experiment, will create a file inside out-dir')
    ## other
    parser.add_argument('--print-iter', default=200, type=int, help='print frequency')
    parser.add_argument('--save-iter', default=50, type=int, help='save frequency')
    parser.add_argument('--eval-iter', default=1000, type=int, help='evaluation frequency')
    parser.add_argument('--seed', default=123, type=int, help='seed for initializing training.')
    
    parser.add_argument('--vis-gt', action='store_true', help='whether visualize GT motions')
    parser.add_argument('--nb-vis', default=20, type=int, help='nb of visualizations')
    
    return parser.parse_args()