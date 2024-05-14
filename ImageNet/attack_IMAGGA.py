from __future__ import print_function
import torch
import numpy as np
import argparse
import os
import torch.nn as nn
import torchvision.models as models

from attack.GAA_to_imagga import GAA_to_IMAGGA
from attack.GA_BASES_to_IMAGGA import GA_BASES_to_IMAGGA

from ImageNet.logger import _get_logger

from ImageNet.utils import Normalize, Interpolate
from ImageNet.Selected_Imagenet_to_Attack_v2 import SelectedImagenet2AttackV2

parser = argparse.ArgumentParser(description='PyTorch ImageNet Test')
parser.add_argument('--model-type', default='resnet50', type=str)
parser.add_argument('--ensemble-surrogate1', action='store_false', default=True,
                    help='VGG16,ResNet18,Squeezenet,Googlenet')
parser.add_argument('--ensemble-surrogate2', action='store_true', default=False,
                    help='VGG19,ResNet50,Inception-v3,MobileNet-v2')
parser.add_argument('--victim-model-type', default='vgg19', type=str)
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--data-dir', default='../../../data/imagenet-selected', type=str)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--model-dir', default='../../saved_model',
                    help='directory of model for saving checkpoint')

args = parser.parse_args()

for arg in vars(args):
    print(arg, ':', getattr(args, arg))

# settings
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda:"+str(args.gpu) if use_cuda else "cpu")
kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}

# GAA_to_IMAGGA Linf
def Untarget_GAA_to_IMAGGA_Linf_test(surrogate_model_list, device, test_loader):
    # GAA_to_IMAGGA中的参数
    epsilon = 12/255
    alpha = 1.2/255
    norm = 'linf'
    steps = 10
    decay = 1.0
    loss_type = 'GAL'
    max_queries = 5
    api_key = 'acc_561050984c7cef0'
    api_secret = '249513df64abc371704bce8a6103f9f2'
    label_projection = np.load('saved_model/label_projection.npy', allow_pickle=True).item()
    logger = _get_logger('logs/GAA_to_IMAGGA_Linf.log', 'info')
    logger.info('---------------------------------------------------')
    if os.path.exists('saved_model/GAA_to_IMAGGA_Linf_success_idx_list_is_logits_False.npy'):
        success_idx_list = np.load('saved_model/GAA_to_IMAGGA_Linf_success_idx_list_is_logits_False.npy').tolist()
    else:
        success_idx_list = []
    if os.path.exists('saved_model/GAA_to_IMAGGA_Linf_idx_list_is_logits_False.npy'):
        idx_list = np.load('saved_model/GAA_to_IMAGGA_Linf_idx_list_is_logits_False.npy').tolist()
    else:
        idx_list = []
    if os.path.exists('saved_model/GAA_to_IMAGGA_Linf_success_query_list_is_logits_False.npy'):
        success_query_list = np.load('saved_model/GAA_to_IMAGGA_Linf_success_query_list_is_logits_False.npy').tolist()
    else:
        success_query_list = []
    if os.path.exists('saved_model/GAA_to_IMAGGA_Linf_success_linf_list_is_logits_False.npy'):
        success_linf_list = np.load('saved_model/GAA_to_IMAGGA_Linf_success_linf_list_is_logits_False.npy').tolist()
    else:
        success_linf_list = []

    gaa_to_imagga = GAA_to_IMAGGA(surrogate_model_list, eps=epsilon, alpha=alpha, steps=steps, decay=decay, loss_type=loss_type, norm=norm, victim_model='imagga', max_queries=max_queries, api_key=api_key, api_secret=api_secret, label_projection=label_projection)
    idx = 0
    for data, target in test_loader:
        if idx < len(idx_list):
            idx += 1
            # print(idx)
            continue
        data, target = data.to(device), target.to(device)
        logger.info('The '+str(idx)+'-th input is being attacked !')
        success, _, n_query, L2, Linf = gaa_to_imagga(data, target, idx, logger)
        if success:
            success_idx_list.append(idx)
            success_query_list.append(n_query)
            success_linf_list.append(Linf)
        idx_list.append(idx)
        idx += 1

        # save status
        np.save('saved_model/GAA_to_IMAGGA_Linf_success_idx_list_is_logits_False.npy', success_idx_list)
        np.save('saved_model/GAA_to_IMAGGA_Linf_idx_list_is_logits_False.npy', idx_list)
        np.save('saved_model/GAA_to_IMAGGA_Linf_success_query_list_is_logits_False.npy', success_query_list)
        np.save('saved_model/GAA_to_IMAGGA_Linf_success_linf_list_is_logits_False.npy', success_linf_list)

    print(
        'GAA_to_IMAGGA bound: {} with {} surrogate(s) is_target: {} epsilon: {} Test: ASR: {}/{} ({:.2f}%), avg_query: {}, median_query: {}, avg_linf: {}'.format(
            norm, len(surrogate_model_list), False, epsilon, len(success_idx_list), len(idx_list), len(success_idx_list)/len(idx_list)*100., np.mean(success_query_list), np.median(success_query_list),
            np.mean(success_linf_list)
        ))

# GAA_to_IMAGGA L2
def Untarget_GAA_to_IMAGGA_L2_test(surrogate_model_list, device, test_loader):
    # GAA_to_IMAGGA中的参数
    epsilon = np.sqrt((1e-3) * 3 * 224 * 224)
    alpha = 2.0
    norm = 'l2'
    steps = 10
    decay = 1.0
    loss_type = 'GAL'
    max_queries = 5
    api_key = 'acc_561050984c7cef0'
    api_secret = '249513df64abc371704bce8a6103f9f2'
    label_projection = np.load('saved_model/label_projection.npy', allow_pickle=True).item()
    logger = _get_logger('logs/GAA_to_IMAGGA_L2.log', 'info')
    logger.info('---------------------------------------------------')
    if os.path.exists('saved_model/GAA_to_IMAGGA_L2_success_idx_list_is_logits_False.npy'):
        success_idx_list = np.load('saved_model/GAA_to_IMAGGA_L2_success_idx_list_is_logits_False.npy').tolist()
    else:
        success_idx_list = []
    if os.path.exists('saved_model/GAA_to_IMAGGA_L2_idx_list_is_logits_False.npy'):
        idx_list = np.load('saved_model/GAA_to_IMAGGA_L2_idx_list_is_logits_False.npy').tolist()
    else:
        idx_list = []
    if os.path.exists('saved_model/GAA_to_IMAGGA_L2_success_query_list_is_logits_False.npy'):
        success_query_list = np.load('saved_model/GAA_to_IMAGGA_L2_success_query_list_is_logits_False.npy').tolist()
    else:
        success_query_list = []
    if os.path.exists('saved_model/GAA_to_IMAGGA_L2_success_l2_list_is_logits_False.npy'):
        success_l2_list = np.load('saved_model/GAA_to_IMAGGA_L2_success_l2_list_is_logits_False.npy').tolist()
    else:
        success_l2_list = []

    gaa_to_imagga = GAA_to_IMAGGA(surrogate_model_list, eps=epsilon, alpha=alpha, steps=steps, decay=decay, loss_type=loss_type, norm=norm, victim_model='imagga', max_queries=max_queries, api_key=api_key, api_secret=api_secret, label_projection=label_projection)
    idx = 0
    for data, target in test_loader:
        if idx < len(idx_list):
            idx += 1
            # print(idx)
            continue
        data, target = data.to(device), target.to(device)
        logger.info('The '+str(idx)+'-th input is being attacked !')
        success, _, n_query, L2, Linf = gaa_to_imagga(data, target, idx, logger)
        if success:
            success_idx_list.append(idx)
            success_query_list.append(n_query)
            success_l2_list.append(L2)
        idx_list.append(idx)
        idx += 1

        # save status
        np.save('saved_model/GAA_to_IMAGGA_L2_success_idx_list_is_logits_False.npy', success_idx_list)
        np.save('saved_model/GAA_to_IMAGGA_L2_idx_list_is_logits_False.npy', idx_list)
        np.save('saved_model/GAA_to_IMAGGA_L2_success_query_list_is_logits_False.npy', success_query_list)
        np.save('saved_model/GAA_to_IMAGGA_L2_success_l2_list_is_logits_False.npy', success_l2_list)

    print(
        'GAA_to_IMAGGA bound: {} with {} surrogate(s) is_target: {} epsilon: {} Test: ASR: {}/{} ({:.2f}%), avg_query: {}, median_query: {}, avg_l2: {}'.format(
            norm, len(surrogate_model_list), False, epsilon, len(success_idx_list), len(idx_list), len(success_idx_list)/len(idx_list)*100., np.mean(success_query_list), np.median(success_query_list),
            np.mean(success_l2_list)
        ))

# GA_BASES_to_IMAGGA Linf
def Untarget_GA_BASES_to_IMAGGA_Linf_test(surrogate_model_list, device, test_loader):
    # GA_BASES_to_IMAGGA中的参数
    epsilon = 12/255
    alpha = 1.2/255
    max_query = 5
    api_key = 'acc_b583055a6cf1cf1'
    api_secret = '483522278868c8b349d205c8ed664300'
    label_projection = np.load('saved_model/label_projection.npy', allow_pickle=True).item()
    logger = _get_logger('logs/GA_BASES_to_IMAGGA_Linf.log', 'info')
    logger.info('---------------------------------------------------')
    if os.path.exists('saved_model/GA_BASES_to_IMAGGA_Linf_success_idx_list_is_logits_False_fuse_loss.npy'):
        success_idx_list = np.load('saved_model/GA_BASES_to_IMAGGA_Linf_success_idx_list_is_logits_False_fuse_loss.npy').tolist()
    else:
        success_idx_list = []
    if os.path.exists('saved_model/GA_BASES_to_IMAGGA_Linf_idx_list_is_logits_False_fuse_loss.npy'):
        idx_list = np.load('saved_model/GA_BASES_to_IMAGGA_Linf_idx_list_is_logits_False_fuse_loss.npy').tolist()
    else:
        idx_list = []
    if os.path.exists('saved_model/GA_BASES_to_IMAGGA_Linf_success_query_list_is_logits_False_fuse_loss.npy'):
        success_query_list = np.load('saved_model/GA_BASES_to_IMAGGA_Linf_success_query_list_is_logits_False_fuse_loss.npy').tolist()
    else:
        success_query_list = []
    if os.path.exists('saved_model/GA_BASES_to_IMAGGA_Linf_success_linf_list_is_logits_False_fuse_loss.npy'):
        success_linf_list = np.load('saved_model/GA_BASES_to_IMAGGA_Linf_success_linf_list_is_logits_False_fuse_loss.npy').tolist()
    else:
        success_linf_list = []

    ga_bases_to_imagga = GA_BASES_to_IMAGGA(surrogate_model_list, n_wb=4, bound='linf', eps=epsilon, n_iters=10, algo='mim', fuse='loss', loss_name='gace', alpha=alpha, lr=5e-3, iterw=max_query, api_key=api_key, api_secret=api_secret, label_projection=label_projection)
    idx = 0
    for data, target in test_loader:
        if idx < len(idx_list):
            idx += 1
            # print(idx)
            continue
        data, target = data.to(device), target.to(device)
        logger.info('The '+str(idx)+'-th input is being attacked !')
        success, _, n_query, L2, Linf = ga_bases_to_imagga(data, target, idx, logger)
        if success:
            success_idx_list.append(idx)
            success_query_list.append(n_query)
            success_linf_list.append(Linf)
        idx_list.append(idx)
        idx += 1

        # save status
        np.save('saved_model/GA_BASES_to_IMAGGA_Linf_success_idx_list_is_logits_False_fuse_loss.npy', success_idx_list)
        np.save('saved_model/GA_BASES_to_IMAGGA_Linf_idx_list_is_logits_False_fuse_loss.npy', idx_list)
        np.save('saved_model/GA_BASES_to_IMAGGA_Linf_success_query_list_is_logits_False_fuse_loss.npy', success_query_list)
        np.save('saved_model/GA_BASES_to_IMAGGA_Linf_success_linf_list_is_logits_False_fuse_loss.npy', success_linf_list)

    print('GA_BASES_to_IMAGGA bound: {} with {} surrogate(s) is_target: {} epsilon: {} Test: ASR: {}/{} ({:.2f}%), avg_query: {}, median_query: {}, avg_linf: {}'.format(
        'linf', len(surrogate_model_list), False, epsilon, len(success_idx_list), len(idx_list),
        len(success_idx_list) / len(idx_list) * 100., np.mean(success_query_list), np.median(success_query_list),
        np.mean(success_linf_list)
    ))

# GA_BASES_to_IMAGGA L2
def Untarget_GA_BASES_to_IMAGGA_L2_test(surrogate_model_list, device, test_loader):
    # GA_BASES_to_IMAGGA中的参数
    epsilon = np.sqrt((1e-3) * 3 * 224 * 224)
    alpha = 2.0
    max_query = 5
    api_key = 'acc_b583055a6cf1cf1'
    api_secret = '483522278868c8b349d205c8ed664300'
    label_projection = np.load('saved_model/label_projection.npy', allow_pickle=True).item()
    logger = _get_logger('logs/GA_BASES_to_IMAGGA_L2.log', 'info')
    logger.info('---------------------------------------------------')
    if os.path.exists('saved_model/GA_BASES_to_IMAGGA_L2_success_idx_list_is_logits_False_fuse_loss.npy'):
        success_idx_list = np.load('saved_model/GA_BASES_to_IMAGGA_L2_success_idx_list_is_logits_False_fuse_loss.npy').tolist()
    else:
        success_idx_list = []
    if os.path.exists('saved_model/GA_BASES_to_IMAGGA_L2_idx_list_is_logits_False_fuse_loss.npy'):
        idx_list = np.load('saved_model/GA_BASES_to_IMAGGA_L2_idx_list_is_logits_False_fuse_loss.npy').tolist()
    else:
        idx_list = []
    if os.path.exists('saved_model/GA_BASES_to_IMAGGA_L2_success_query_list_is_logits_False_fuse_loss.npy'):
        success_query_list = np.load('saved_model/GA_BASES_to_IMAGGA_L2_success_query_list_is_logits_False_fuse_loss.npy').tolist()
    else:
        success_query_list = []
    if os.path.exists('saved_model/GA_BASES_to_IMAGGA_L2_success_l2_list_is_logits_False_fuse_loss.npy'):
        success_l2_list = np.load('saved_model/GA_BASES_to_IMAGGA_L2_success_l2_list_is_logits_False_fuse_loss.npy').tolist()
    else:
        success_l2_list = []

    ga_bases_to_imagga = GA_BASES_to_IMAGGA(surrogate_model_list, n_wb=4, bound='l2', eps=epsilon, n_iters=10, algo='mim', fuse='loss', loss_name='gace', alpha=alpha, lr=5e-3, iterw=max_query, api_key=api_key, api_secret=api_secret, label_projection=label_projection)
    idx = 0
    for data, target in test_loader:
        if idx < len(idx_list):
            idx += 1
            # print(idx)
            continue
        data, target = data.to(device), target.to(device)
        logger.info('The '+str(idx)+'-th input is being attacked !')
        success, _, n_query, L2, Linf = ga_bases_to_imagga(data, target, idx, logger)
        if success:
            success_idx_list.append(idx)
            success_query_list.append(n_query)
            success_l2_list.append(L2)
        idx_list.append(idx)
        idx += 1

        # save status
        np.save('saved_model/GA_BASES_to_IMAGGA_L2_success_idx_list_is_logits_False_fuse_loss.npy', success_idx_list)
        np.save('saved_model/GA_BASES_to_IMAGGA_L2_idx_list_is_logits_False_fuse_loss.npy', idx_list)
        np.save('saved_model/GA_BASES_to_IMAGGA_L2_success_query_list_is_logits_False_fuse_loss.npy', success_query_list)
        np.save('saved_model/GA_BASES_to_IMAGGA_L2_success_l2_list_is_logits_False_fuse_loss.npy', success_l2_list)

    print('GA_BASES_to_IMAGGA bound: {} with {} surrogate(s) is_target: {} epsilon: {} Test: ASR: {}/{} ({:.2f}%), avg_query: {}, median_query: {}, avg_l2: {}'.format(
        'l2', len(surrogate_model_list), False, epsilon, len(success_idx_list), len(idx_list),
        len(success_idx_list) / len(idx_list) * 100., np.mean(success_query_list), np.median(success_query_list),
        np.mean(success_l2_list)
    ))

def test_main():
    # 选择使得所有模型都分类正确的对抗样本
    predict_status_arr = np.loadtxt(os.path.join(args.model_dir, 'predict_status_arr_v2.txt'))
    predict_status_arr = (predict_status_arr != 0)

    # 加载替代模型和受害者模型都分类正确的测试样本集
    testset = SelectedImagenet2AttackV2(predict_status_arr, args.data_dir)
    test_loader_1 = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, pin_memory=True,
                                              num_workers=0)

    # 加载替代模型
    if args.model_type == 'resnet50' and not args.ensemble_surrogate1 and not args.ensemble_surrogate2:
        model = models.resnet50()
        model.load_state_dict(torch.load(os.path.join(args.model_dir, 'resnet50_checkpoint.pth')))
        model = nn.Sequential(
            Normalize(),
            model
        )
        model.to(device)
        model.eval()
        model_list = [model]
    elif args.model_type == 'resnet18' and not args.ensemble_surrogate1 and not args.ensemble_surrogate2:
        model_resnet18 = models.resnet18()
        model_resnet18.load_state_dict(torch.load(os.path.join(args.model_dir, 'resnet18_checkpoint.pth')))
        model_resnet18 = nn.Sequential(
            Normalize(),
            model_resnet18
        )
        model_resnet18.to(device)
        model_resnet18.eval()
        model_list = [model_resnet18]
    elif args.ensemble_surrogate1:
        model_vgg16 = models.vgg16()
        model_vgg16.load_state_dict(torch.load(os.path.join(args.model_dir, 'vgg16_checkpoint.pth')))
        model_vgg16 = nn.Sequential(
            Normalize(),
            model_vgg16
        )
        model_vgg16.to(device)
        model_vgg16.eval()

        model_resnet18 = models.resnet18()
        model_resnet18.load_state_dict(torch.load(os.path.join(args.model_dir, 'resnet18_checkpoint.pth')))
        model_resnet18 = nn.Sequential(
            Normalize(),
            model_resnet18
        )
        model_resnet18.to(device)
        model_resnet18.eval()

        model_squeezenet = models.squeezenet1_1()
        model_squeezenet.load_state_dict(torch.load(os.path.join(args.model_dir, 'squeezenet1_1_checkpoint.pth')))
        model_squeezenet = nn.Sequential(
            Normalize(),
            model_squeezenet
        )
        model_squeezenet.to(device)
        model_squeezenet.eval()

        model_googlenet = models.googlenet()
        model_googlenet.load_state_dict(torch.load(os.path.join(args.model_dir, 'googlenet_checkpoint.pth')))
        model_googlenet = nn.Sequential(
            Normalize(),
            model_googlenet
        )
        model_googlenet.to(device)
        model_googlenet.eval()

        model_list = [model_vgg16, model_resnet18, model_squeezenet, model_googlenet]
    elif args.ensemble_surrogate2:
        model_vgg19 = models.vgg19()
        model_vgg19.load_state_dict(torch.load(os.path.join(args.model_dir, 'vgg19_checkpoint.pth')))
        model_vgg19 = nn.Sequential(
            Normalize(),
            model_vgg19
        )
        model_vgg19.to(device)
        model_vgg19.eval()

        model_resnet50 = models.resnet50()
        model_resnet50.load_state_dict(torch.load(os.path.join(args.model_dir, 'resnet50_checkpoint.pth')))
        model_resnet50 = nn.Sequential(
            Normalize(),
            model_resnet50
        )
        model_resnet50.to(device)
        model_resnet50.eval()

        model_inceptionv3 = models.Inception3(init_weights=True)
        model_inceptionv3.load_state_dict(torch.load(os.path.join(args.model_dir, 'inception_v3_checkpoint.pth')))
        model_inceptionv3 = nn.Sequential(
            Interpolate(torch.Size([299, 299]), 'bilinear'),
            Normalize(),
            model_inceptionv3
        )
        model_inceptionv3.to(device)
        model_inceptionv3.eval()

        model_mobilenetv2 = models.mobilenet_v2()
        model_mobilenetv2.load_state_dict(torch.load(os.path.join(args.model_dir, 'mobilenet_v2_checkpoint.pth')))
        model_mobilenetv2 = nn.Sequential(
            Normalize(),
            model_mobilenetv2
        )
        model_mobilenetv2.to(device)
        model_mobilenetv2.eval()

        model_list = [model_vgg19, model_resnet50, model_inceptionv3, model_mobilenetv2]

    # Linf attack setting
    Untarget_GAA_to_IMAGGA_Linf_test(model_list, device, test_loader_1)

    Untarget_GA_BASES_to_IMAGGA_Linf_test(model_list, device, test_loader_1)

    # L2 attack setting
    Untarget_GAA_to_IMAGGA_L2_test(model_list, device, test_loader_1)

    Untarget_GA_BASES_to_IMAGGA_L2_test(model_list, device, test_loader_1)

if __name__ == '__main__':
    test_main()