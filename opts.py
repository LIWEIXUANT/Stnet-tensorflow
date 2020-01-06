import argparse
parser = argparse.ArgumentParser(description='Tensorflow implement of Temporal Segment Network')
parser.add_argument('--dataset', type=str, default='ucf101')
parser.add_argument('--modality', type=str, default='RGB')
parser.add_argument('--train_list', type=str, default='')
parser.add_argument('--val_list', type=str, default='')
parser.add_argument('--root_path', type=str, default='')
parser.add_argument('--store_name', type=str, default='')
# ============================ Model Configs ===========================
parser.add_argument('--arch', type=str, default='resnet50')
parser.add_argument('--num_segment', type=int, default=8)
parser.add_argument('--consensus_type', type=str, default='TRNmultiscale')
parser.add_argument('--k', type=int, default=3)

parser.add_argument('--dropout', '--do', default=0.8, type=float)
parser.add_argument('--loss_type', type=str, default='nll', choices=['nll'])
parser.add_argument('--img_feature_dim', default=256, type=int)
parser.add_argument('--checkpoint_file', default='./pretrained_weights/weights_resnet.npy', type=str)

# ========================== Learning Configs ==========================
parser.add_argument('--epochs', default=130, type=int)
parser.add_argument('-b', '--batch_size', default=1, type=int)
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--lr_steps', default=[50, 100], type=float, nargs='+', help='epochs to decay learning rate by 10')
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--weight-decay', default=5e-4, type=float)
parser.add_argument('--clip-gredient', default=20, type=float, help='gradient norm clipping')
parser.add_argument('--no_partialbn', '-npb', default=False, action='store_true')

# ========================== Monitor Configs ==========================
parser.add_argument('--print-freq', '-p', default=20, type=int)
parser.add_argument('--eval-freq', default=1, type=int)

# ========================== Runtime Configs ==========================
parser.add_argument('--root_log', type=str, default='log')
parser.add_argument('--root_model', type=str, default='model')
parser.add_argument('--root_checkpoint', type=str, default='checkpiont')

