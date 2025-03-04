import argparse
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn.functional import mse_loss

from data.datasets import get_dataloader
from models.student_net import build_student_net
# from models.student_net_c11 import build_student_net
# from models.student_net_l import build_student_net
from utils import false_alarm_rate, Result_Meter, stf, stf2


def get_args() -> argparse.Namespace:
    """parser args"""
    parser = argparse.ArgumentParser(description='BSDM training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset options
    # parser.add_argument('--train_dir', type=str, default='/home/worker1/DATASETS/HAD100/mat/without_anomaly/', help='Folder to train datasets.')
    parser.add_argument('--test_dir', type=str, default='/home/worker1/DATASETS/HAD100/mat/with_anomaly/', help='Folder to test datasets.')
    # parser.add_argument('--test_dir', type=str, default='/home/worker1/DATASETS/TH-Datasets/', help='Folder to test datasets.')
    # parser.add_argument('--test_dir', type=str, default='/media/worker1/一栋硬盘/HSI-D-Datasets/Others/', help='Folder to test datasets.')
    # parser.add_argument('--test_dir', type=str, default='/home/worker1/DATASETS/abu/', help='Folder to test datasets.')
    # parser.add_argument('--test_dir', type=str, default='/home/worker1/DATASETS/西电一号数据/异常检测-TH03/', help='Folder to test datasets.')
    parser.add_argument('--channels', type=int, default=50, help='Number of bands for input data.')

    # Training options
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.0001, help='The Initial Learning Rate.')
    parser.add_argument('--interval', type=int, default=10, help='Interval of output accuracy.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size of train set.')

    # Random seed
    parser.add_argument('--seed', type=int, default=12345, help='Manual seed.')

    # Device to use
    parser.add_argument('--device', type=str, default='gpu', choices=['cpu', 'gpu'], help='Device to use.')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU to use.')

    return parser.parse_args()


def validate(trained_model, dataloader, device):
    # trained_model = trained_model.half()

    trained_model.eval()
    results_meter = Result_Meter(['filename', 'auc', 'fpr', 'time'])
    for index, (data, label, filename) in enumerate(dataloader):
        # if filename[0] == 'ang20170821t183707_104.mat':
        # if "HYDICE" in filename[0]:
            # data = data.half()
            s_time = time.time()

            data = data.to(device)
            data.requires_grad_()
            label = label.to(device)

            s_out_list = trained_model(data)

            mask, rx_out = stf(data.detach().cpu().squeeze().numpy().transpose(1, 2, 0), device, True)
            loss = mse_loss(s_out_list[-1], data, reduce=False)
            for s_out in s_out_list[0:-1]:
                loss += mse_loss(s_out, data, reduce=False)

            loss = torch.mean(loss * mask)
            loss.backward()

            saliency = torch.abs(data.grad)
            saliency = np.transpose(saliency.detach().cpu().numpy(), (0, 2, 3, 1)).squeeze()
            saliency = np.max(saliency, axis=-1)

            # saliency = torch.mean(loss * mask, dim=[0, 1]).detach().cpu().numpy()

            time_cost = time.time() - s_time
            results_meter.results['time'].append(time_cost)

            label = np.squeeze(label.cpu().numpy())

            auc, fpr, _ = false_alarm_rate(label.reshape([-1]), saliency.reshape([-1]))
            print(filename[0], auc, fpr, time_cost)

            results_meter.results['filename'].append(filename)
            results_meter.results['auc'].append(auc)
            results_meter.results['fpr'].append(fpr)

            auc, fpr, _ = false_alarm_rate(label.reshape([-1]), rx_out.reshape([-1]))
            print('rx: ', auc, fpr, time_cost)

            # plt.figure()
            # plt.imshow(saliency.T, cmap='jet')
            # plt.show()

    avg_auc = results_meter.avg('auc')
    avg_fpr = results_meter.avg('fpr')
    avg_time = results_meter.avg('time')
    print(avg_auc, avg_fpr, avg_time)
    trained_model.zero_grad()


def main():
    # ger args
    args = get_args()

    # set device
    device = args.device
    if device == 'gpu':
        device = args.gpu_id

    # load data
    test_dataloader = get_dataloader(root=args.test_dir, channels=args.channels, with_label=True, shuffle=False, transform=False)

    s_model = build_student_net(args.channels).to(device)
    # state_dict = torch.load('./results/student_net_c11_best.pt')['state_dict']
    state_dict = torch.load('./results/student_net_best.pt')['state_dict']
    s_model.load_state_dict(state_dict)
    s_model.eval()

    validate(trained_model=s_model, dataloader=test_dataloader, device=device)


if __name__ == '__main__':
    main()
