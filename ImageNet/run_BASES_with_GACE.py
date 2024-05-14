from __future__ import print_function
import torch
import numpy as np
import argparse
import os
import time
import torch.nn as nn
import torchvision.models as models
import timm

from attack.GA_BASES import GA_BASES

from ImageNet.utils import Normalize, Interpolate
from ImageNet.Selected_Imagenet_to_Attack import SelectedImagenet2Attack

parser = argparse.ArgumentParser(description='PyTorch ImageNet Test')
parser.add_argument('--model-type', default='resnet50', type=str)
parser.add_argument('--ensemble-surrogate1', action='store_true', default=False,
                    help='VGG16,ResNet18,Squeezenet,Googlenet')
parser.add_argument('--ensemble-surrogate2', action='store_true', default=False,
                    help='VGG19,ResNet50,Inception-v3,MobileNet-v2')
parser.add_argument('--victim-model-type', default='vgg19', type=str)
parser.add_argument('--test-batch-size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--max-query', default=50, type=int)
parser.add_argument('--data-dir', default='../data/', type=str)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--gpu', default=0, type=int)
parser.add_argument('--model-dir', default='./saved_model',
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
device = torch.device("cuda:" + str(args.gpu) if use_cuda else "cpu")
kwargs = {'num_workers': 0, 'pin_memory': True} if use_cuda else {}


# GA_BASES
def GA_BASES_Test(surrogate_model_list, victim_model, epsilon, bound, is_target, device, test_loader):
    # GA_BASES中的参数
    max_query = 50
    success_list = []
    query_list = []
    l2_list = []
    linf_list = []
    bases = GA_BASES(victim_model, surrogate_model_list, n_wb=4, bound=bound, eps=epsilon, n_iters=10, algo='mim',
                     fuse='loss', loss_name='gace', times_alpha=1, lr=5e-3, iterw=max_query)
    if is_target:
        bases.set_mode_targeted_random()
    start_time = time.time()
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        if is_target:
            target_labels = bases._get_target_label(data, target)
            success, _, n_query, L2, Linf = bases(data, target_labels)
        else:
            success, _, n_query, L2, Linf = bases(data, target)
        success_list.append(success)
        query_list.append(n_query)
        l2_list.append(L2)
        linf_list.append(Linf)

    success_arr = np.array(success_list)
    query_arr = np.array(query_list)
    l2_arr = np.array(l2_list)
    linf_arr = np.array(linf_list)

    for max_query_temp in np.arange(10, max_query + 1, 10):
        success_count = (success_arr & (query_arr < (max_query_temp + 1))).sum()
        total_count = len(success_arr)
        ASR = success_count / total_count * 100.
        avg_query = query_arr[success_arr & (query_arr < (max_query_temp + 1))].mean()
        median_query = np.median(query_arr[success_arr & (query_arr < (max_query_temp + 1))])
        avg_l2 = l2_arr[success_arr & (query_arr < (max_query_temp + 1))].mean()
        avg_linf = linf_arr[success_arr & (query_arr < (max_query_temp + 1))].mean()

        print(
            'GA_BASES bound: {} with {} surrogate(s) is_target: {} epsilon: {} Test: ASR: {}/{} ({:.2f}%), avg_query: {}, median_query: {}, avg_l2: {}, avg_linf: {}'.format(
                bound, len(surrogate_model_list), is_target, epsilon, success_count, total_count, ASR, avg_query,
                median_query, avg_l2, avg_linf
            ))
    end_time = time.time()
    avg_time_spended = (end_time - start_time) / 60
    print('avg_time_spended:{}'.format(avg_time_spended))

    return success_arr, query_arr, l2_arr, linf_arr

def test_main():
    # 选择使得所有模型都分类正确的对抗样本
    predict_status_arr = np.loadtxt(os.path.join(args.model_dir, 'predict_status_arr_v2.txt'))
    predict_status_arr = (predict_status_arr != 0)

    # 加载替代模型和受害者模型都分类正确的测试样本集
    testset = SelectedImagenet2Attack(predict_status_arr, args.data_dir)
    test_loader_1 = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False,
                                                pin_memory=True,
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

    # 加载受害者模型
    if args.victim_model_type == 'vgg16':
        victim_model = models.vgg16()
        victim_model.load_state_dict(torch.load(os.path.join(args.model_dir, 'vgg16_checkpoint.pth')))
        victim_model = nn.Sequential(
            Normalize(),
            victim_model
        )
        victim_model.to(device)
        victim_model.eval()
    elif args.victim_model_type == 'vgg19':
        victim_model = models.vgg19()
        victim_model.load_state_dict(torch.load(os.path.join(args.model_dir, 'vgg19_checkpoint.pth')))
        victim_model = nn.Sequential(
            Normalize(),
            victim_model
        )
        victim_model.to(device)
        victim_model.eval()
    elif args.victim_model_type == 'resnet152':
        victim_model = models.resnet152()
        victim_model.load_state_dict(torch.load(os.path.join(args.model_dir, 'resnet152_checkpoint.pth')))
        victim_model = nn.Sequential(
            Normalize(),
            victim_model
        )
        victim_model.to(device)
        victim_model.eval()
    elif args.victim_model_type == 'densenet121':
        victim_model = models.densenet121(pretrained=True)
        # victim_model.load_state_dict(
        #     torch.load(os.path.join(args.model_dir, 'densenet121_checkpoint.pth')))
        victim_model = nn.Sequential(
            Normalize(),
            victim_model
        )
        victim_model.to(device)
        victim_model.eval()
    elif args.victim_model_type == 'Ens_Adv_Inception_ResNet_v2':
        victim_model = timm.create_model('ens_adv_inception_resnet_v2', pretrained=True)
        victim_model = nn.Sequential(
            Interpolate(torch.Size([299, 299]), 'bilinear'),
            Normalize(),
            victim_model
        )
        victim_model.to(device)
        victim_model.eval()

    # L2攻击+Untarget
    epsilon = np.sqrt((1e-3) * 3 * 224 * 224)
    bound = 'l2'
    is_target = False

    if args.ensemble_surrogate1 or args.ensemble_surrogate2:
        infos_ga_bases = GA_BASES_Test(model_list, victim_model, epsilon, bound, is_target, device, test_loader_1)
        np.save('saved_model/EnsSurro1_' + str(args.ensemble_surrogate1) + '_EnsSurro2_' + str(
            args.ensemble_surrogate2) +'_SingleSurro_'+args.model_type+ '_Victim_' + args.victim_model_type + '_maxQuery_' + str(
            50) + 'epsilon_12.27_GA_BASES_L2_Untarget_w_wo_GACE_v2_L2.npy', infos_ga_bases)

    # L2攻击+Target
    epsilon = np.sqrt((1e-3) * 3 * 224 * 224) * 2
    bound = 'l2'
    is_target = True

    if args.ensemble_surrogate1 or args.ensemble_surrogate2:
        infos_ga_bases = GA_BASES_Test(model_list, victim_model, epsilon, bound, is_target, device, test_loader_1)
        np.save('saved_model/EnsSurro1_' + str(args.ensemble_surrogate1) + '_EnsSurro2_' + str(
            args.ensemble_surrogate2) +'_SingleSurro_'+args.model_type+ '_Victim_' + args.victim_model_type + '_maxQuery_' + str(
            50) + 'epsilon_24.54_GA_BASES_L2_Target_w_wo_GACE_v2_L2.npy', infos_ga_bases)


if __name__ == '__main__':
    test_main()
