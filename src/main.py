import os
import random
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import logging
import time
from model import create_model_diffu, Att_Diffuse_model
from trainer import model_train
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


parser = argparse.ArgumentParser()
parser.add_argument('--s_dataset', default='movie')
parser.add_argument('--t_dataset', default='video')
parser.add_argument('--p', type=float, default=0.3)
parser.add_argument('--w', type=float, default=1.0)
parser.add_argument('--log_file', default='log/', help='log dir path')
parser.add_argument('--random_seed', type=int, default=1997, help='Random seed')  
parser.add_argument('--max_len', type=int, default=8, help='The max length of sequence')
parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--num_gpu', type=int, default=1, help='Number of GPU')
parser.add_argument('--batch_size', type=int, default=512, help='Batch Size')  
parser.add_argument("--hidden_size", default=128, type=int, help="hidden size of model")
parser.add_argument('--dropout', type=float, default=0.1, help='Dropout of representation')
parser.add_argument('--emb_dropout', type=float, default=0.3, help='Dropout of item embedding')
parser.add_argument("--hidden_act", default="gelu", type=str) # gelu relu
parser.add_argument('--num_blocks', type=int, default=4, help='Number of Transformer blocks')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs for training') 
parser.add_argument('--decay_step', type=int, default=100, help='Decay step for StepLR')
parser.add_argument('--gamma', type=float, default=0.1, help='Gamma for StepLR')
parser.add_argument('--metric_ks', nargs='+', type=int, default=[5, 10, 20], help='ks for Metric@k')
parser.add_argument('--optimizer', type=str, default='Adam', choices=['SGD', 'Adam'])
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--loss_lambda', type=float, default=1, help='loss weight for diffusion')
parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization')
parser.add_argument('--momentum', type=float, default=None, help='SGD momentum')
parser.add_argument('--schedule_sampler_name', type=str, default='lossaware', help='Diffusion for t generation')
parser.add_argument('--diffusion_steps', type=int, default=32, help='Diffusion step')
parser.add_argument('--lambda_uncertainty', type=float, default=0.001, help='uncertainty weight')
parser.add_argument('--noise_schedule', default='trunc_lin', help='Beta generation')  ## cosine, linear, trunc_cos, trunc_lin, pw_lin, sqrt
parser.add_argument('--rescale_timesteps', default=True, help='rescal timesteps')
parser.add_argument('--eval_interval', type=int, default=5, help='the number of epoch to eval')
parser.add_argument('--patience', type=int, default=2, help='the number of epoch to wait before early stop')
parser.add_argument('--description', type=str, default='Diffu_norm_score', help='Model brief introduction')
args = parser.parse_args()


if not os.path.exists(args.log_file):
    os.makedirs(args.log_file)
if not os.path.exists(args.log_file + args.s_dataset + "_" + args.t_dataset):
    os.makedirs(args.log_file + args.s_dataset + "_" + args.t_dataset )


logging.basicConfig(level=logging.INFO, filename=args.log_file + args.s_dataset + "_" + args.t_dataset + '/' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.log',
                    datefmt='%Y/%m/%d %H:%M:%S', format='%(asctime)s - %(name)s - %(levelname)s - %(lineno)d - %(module)s - %(message)s', filemode='w')
logger = logging.getLogger(__name__)


def fix_random_seed_as(random_seed):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def item_num_create(args, source_item_num, target_item_num):
    args.source_item_num = source_item_num
    args.target_item_num = target_item_num
    args.item_num = source_item_num
    args.pretrain_flag = True
    return args

def main(args):    
    fix_random_seed_as(args.random_seed)
    if args.t_dataset == "toy" or args.t_dataset == "sports":
        args.dropout = 0.3
    print(args)
    logger.info(args)
    path_s_data = '../data/' + args.s_dataset
    s_train_path = path_s_data + '/train.df'
    s_val_path = path_s_data + '/val.df'
    s_test_path = path_s_data + '/test.df'
    path_t_data = '../data/' + args.t_dataset
    t_train_path = path_t_data + '/train.df'
    t_val_path = path_t_data + '/val.df'
    t_test_path = path_t_data + '/test.df'

    
    s_tra_data = pd.read_pickle(s_train_path)
    s_val_data = pd.read_pickle(s_val_path)
    s_test_data = pd.read_pickle(s_test_path)
    t_tra_data = pd.read_pickle(t_train_path)
    t_val_data = pd.read_pickle(t_val_path)
    t_test_data = pd.read_pickle(t_test_path)

    source_item_num = 98507
    target_item_num = 23978

    args = item_num_create(args, source_item_num, target_item_num)
    
    s_val_data['negative_samples'] = None
    m = s_val_data['seq'].size
    for i in range(m):
        negative_sample = []
        negative_sample.append(s_val_data['next'][i])
        while len(negative_sample)<101 :
            sample = random.randint(0, args.source_item_num - 1)
            if sample not in negative_sample:
                negative_sample.append(sample)
        s_val_data.at[i, 'negative_samples'] = negative_sample
    t_val_data['negative_samples'] = None
    m = t_val_data['seq'].size
    for i in range(m):
        negative_sample = []
        negative_sample.append(t_val_data['next'][i])
        while len(negative_sample)<101 :
            sample = random.randint(0, args.target_item_num - 1)
            if sample not in negative_sample:
                negative_sample.append(sample)
        t_val_data.at[i, 'negative_samples'] = negative_sample
    s_test_data['negative_samples'] = None
    m = s_test_data['seq'].size
    for i in range(m):
        negative_sample = []
        negative_sample.append(s_test_data['next'][i])
        while len(negative_sample)<101 :
            sample = random.randint(0, args.source_item_num - 1)
            if sample not in negative_sample:
                negative_sample.append(sample)
        s_test_data.at[i, 'negative_samples'] = negative_sample
    t_test_data['negative_samples'] = None
    m = t_test_data['seq'].size
    for i in range(m):
        negative_sample = []
        negative_sample.append(t_test_data['next'][i])
        while len(negative_sample)<101 :
            sample = random.randint(0, args.target_item_num - 1)
            if sample not in negative_sample:
                negative_sample.append(sample)
        t_test_data.at[i, 'negative_samples'] = negative_sample
        
    diffu_rec = create_model_diffu(args)
    rec_diffu_joint_model = Att_Diffuse_model(diffu_rec, args)
    
    pretrain_flag = True
    best_model, test_results = model_train(s_tra_data, s_val_data, s_test_data, t_tra_data, rec_diffu_joint_model, args, logger, pretrain_flag)
    rec_diffu_joint_model.load_state_dict(torch.load("./saved_model/"+args.s_dataset + "_" + args.t_dataset+"/model.pth"))
    args.item_num = target_item_num
    args.eval_interval = 1
    args.patience = 5
    pretrain_flag = False
    best_model, test_results = model_train(t_tra_data, t_val_data, t_test_data, None, rec_diffu_joint_model, args, logger, pretrain_flag)


if __name__ == '__main__':
    main(args)
