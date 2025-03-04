import copy
import random
import argparse
import numpy as np
import torch
from models.teacher_net import build_teacher_net
from data.datasets import get_dataloader
import matplotlib.pyplot as plt
from matplotlib import colormaps
from torch.nn.functional import mse_loss, cosine_similarity
from scipy.ndimage import filters
import cv2

from utils import normalize, false_alarm_rate, Result_Meter, stf


def get_args() -> argparse.Namespace:
    """parser args"""
    parser = argparse.ArgumentParser(description='STAD training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Dataset options
    parser.add_argument('--train_dir', type=str, default='/home/worker1/DATASETS/HAD100/mat/without_anomaly/', help='Folder to train datasets.')
    parser.add_argument('--test_dir', type=str, default='/home/worker1/DATASETS/HAD100/mat/with_anomaly/', help='Folder to test datasets.')
    # parser.add_argument('--test_dir', type=str, default='/home/worker1/DATASETS/abu/', help='Folder to test datasets.')
    parser.add_argument('--channels', type=int, default=10, help='Number of bands for input data.')

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


def set_seed(seed=None):
    if seed is None:
        set_seed(random.randint(0, 2 ** 30))
    else:
        print('===>>> Seed = {} <<<==='.format(seed))
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def validate(trained_model, dataloader, device):
    trained_model.eval()
    results_meter = Result_Meter(['filename', 'auc', 'fpr'])
    for index, (data, label, filename) in enumerate(dataloader):
        data = data.to(device)
        data.requires_grad_()
        label = label.to(device)

        t_out_list = trained_model.forward(data)

        mask = stf(data.detach().cpu().squeeze().numpy().transpose(1, 2, 0), device)

        loss = mse_loss(t_out_list[-1], data, reduce=False)
        for out in t_out_list[0:-1]:
            loss += mse_loss(out, data, reduce=False)

        loss = torch.mean(loss * mask)

        loss.backward()
        saliency = torch.abs(data.grad)
        saliency = np.transpose(saliency.detach().cpu().numpy(), (0, 2, 3, 1)).squeeze()
        saliency = np.max(saliency, axis=-1)

        label = np.squeeze(label.cpu().numpy())

        auc, fpr, _ = false_alarm_rate(label.reshape([-1]), saliency.reshape([-1]))
        # print(filename[0], auc, fpr)
        results_meter.results['filename'].append(filename)
        results_meter.results['auc'].append(auc)
        results_meter.results['fpr'].append(fpr)
    avg_auc = results_meter.avg('auc')
    avg_fpr = results_meter.avg('fpr')
    print(avg_auc, avg_fpr)
    trained_model.zero_grad()

    return results_meter.avg('auc'), results_meter.avg('fpr')


def update_ema_model(test_model, training_model, decay_rate=0.99):
    for test_param, training_param in zip(test_model.parameters(), training_model.parameters()):
        test_param.data.mul_(decay_rate).add_(training_param.data, alpha=1 - decay_rate)


def main():
    # ger args
    args = get_args()

    # set device
    device = args.device
    if device == 'gpu':
        device = args.gpu_id

    # set seed
    set_seed(args.seed)

    # load data
    train_dataloader = get_dataloader(root=args.train_dir, channels=args.channels, batch_size=args.batch_size, size=64)
    test_dataloader = get_dataloader(root=args.test_dir, channels=args.channels, with_label=True, shuffle=False, transform=False)

    model = build_teacher_net(args.channels, stages=3).to(device)

    ema_model = copy.deepcopy(model)
    ema_model.eval()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    plot_items = {'auc': [], 'fpr': []}

    args.interval = min(args.interval, args.epochs)

    for e in range(args.epochs):
        model.train()
        losses = 0
        length = len(train_dataloader)
        for index, (data, label, _) in enumerate(train_dataloader):
            data = data.to(device)
            label = label.to(device)
            t_out_list = model(data * (label + 1))

            loss = mse_loss(data, t_out_list[-1])
            for t_out in t_out_list[:-1]:
                loss += mse_loss(data, t_out)

            loss.backward()
            opt.step()
            opt.zero_grad()
            losses += loss.item()

            update_ema_model(ema_model, model)

        print('Epoch: {} | loss: {:.4f}'.format(e+1, losses/length))
        if (e+1) % args.interval == 0:
            auc, fpr = validate(trained_model=ema_model, dataloader=test_dataloader, device=device)
            plot_items['auc'].append(auc)
            plot_items['fpr'].append(fpr)
            save_dir = './results_stages_k/teacher_net_e{}.pt'.format(e + 1)
            torch.save({'state_dict': ema_model.state_dict()}, save_dir)

    plt.subplot(1, 2, 1)
    plt.plot(plot_items['auc'])
    plt.subplot(1, 2, 2)
    plt.plot(plot_items['fpr'])
    plt.show()

if __name__ == '__main__':
    main()
