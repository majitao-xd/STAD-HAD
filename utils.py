import cv2
import time
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
# from cv2.ximgproc import guidedFilter
from torch.nn.functional import avg_pool2d
from skimage import transform as sk_transform


def normalize(data):
    return (data - data.min()) / (data.max() - data.min())


# 这个函数是主体部分
def stf(data, device, rx_res=False):
    data = normalize(data)  # 用ｍａｘ和ｍｉｎ实现的，他的算子支持
    M, N, B = data.shape
    data = np.reshape(data, [M * N, B]).T
    mu = np.mean(data, axis=1)
    # sigma = cov(data)   # 这个ＣＡＮＮ５里面没找到，应该是没有，参考下面我写的函数
    sigma = np.cov(data)
    z = data - mu[:, np.newaxis]
    # sig_inv = pinv(sigma)
    sig_inv = np.linalg.inv(sigma)  # 这个ｉｎｖ是有的，但要验证一下数值一部一样，不一样就用我下面写好的ｐｉｎｖ
    rx_out = np.sum(np.matmul(z.T, sig_inv) * z.T, axis=-1)
    rx_out = np.reshape(rx_out, [M, N])

    dist_data = cv2.bilateralFilter(np.array(normalize(rx_out) * 255, dtype=np.float32), d=3, sigmaColor=80, sigmaSpace=0)
    # dist_data = bilateral_filter(np.array(normalize(rx_out) * 255, dtype=np.float32), 1, 80) # 写好了，直接用就行
    dist_data = torch.tensor(dist_data).to(device).unsqueeze(0).unsqueeze(0)
    dist_data = torch.clamp(dist_data - torch.median(dist_data), min=0)

    if rx_res:
        return dist_data, rx_out
    else:
        return dist_data


def stf2(data, device, rx_res=False):
    # data = normalize(data)  # 用ｍａｘ和ｍｉｎ实现的，他的算子支持
    # guide = data.max(axis=-1)
    M, N, B = data.shape
    data = np.reshape(data, [M * N, B]).T
    mu = np.mean(data, axis=1)
    sigma = cov(data)  # 这个ＣＡＮＮ５里面没找到，应该是没有，参考下面我写的函数
    # sigma = np.cov(data)
    z = data - mu[:, np.newaxis]
    # sig_inv = pinv(sigma)
    # sig_inv = np.linalg.inv(sigma)  # 这个ｉｎｖ是有的，但要验证一下数值一部一样，不一样就用我下面写好的ｐｉｎｖ
    sig_inv = sigma
    rx_out = np.sum(np.matmul(z.T, sig_inv) * z.T, axis=-1)
    rx_out = np.reshape(rx_out, [M, N])

    dist_data = guided_filter(rx_out, rx_out, 1, 1e-5)
    # dist_data = bilateral_filter(np.array(normalize(rx_out) * 255, dtype=np.float32), 1, 80)  # 写好了，直接用就行
    # dist_data = torch.tensor(dist_data).to(device).unsqueeze(0).unsqueeze(0)
    dist_data = torch.clamp(dist_data - torch.median(dist_data), min=0)

    # dist_data = torch.nn.functional.max_pool2d(dist_data, 3, 1, 1) + torch.nn.functional.max_pool2d(-dist_data, 3, 1, 1)

    if rx_res:
        return dist_data, rx_out
    else:
        return dist_data


def pinv(x):
    u, s, vh = np.linalg.svd(x)
    s = 1 / s
    res = np.matmul(np.transpose(vh), s[..., np.newaxis] * np.transpose(u))

    return res


def bilateral_filter(img, r, sigma_color):
    img_pad = np.pad(img, ((1, 1), (1, 1)), mode='edge')
    H, W = img.shape
    d = int(r * 2 + 1)

    mask = np.zeros([d, d])
    for i in range(d):
        for j in range(d):
            mask[i, j] = ((i - r)**2 + (j - r)**2)**0.5 <= r

    gauss_color_coe = -0.5 / sigma_color ** 2
    gauss_space_coe = -0.5 / r ** 2

    new_img = np.zeros_like(img)

    X, Y = np.meshgrid(np.linspace(-r, r, d), np.linspace(-r, r, d)) # 这个应该是没有现成的算子，你得重新写一下，但很简单
    gauss_space_kernel = np.exp((X ** 2 + Y ** 2) * gauss_space_coe)

    for i in range(H):
        for j in range(W):
            gauss_color_kernel = np.exp((img_pad[i:i+d, j:j+d] - img_pad[i+r, j+r]) ** 2 * gauss_color_coe)
            weight_coe = gauss_color_kernel * gauss_space_kernel * mask
            weight_n = np.sum(weight_coe * img_pad[i:i+d, j:j+d])
            weight_d = np.sum(weight_coe)
            new_img[i, j] = weight_n / weight_d

    return new_img


def cov(x):
    a = np.array(x, np.float32)
    avg = np.mean(a, axis=1, keepdims=True)
    fact = a.shape[1] - 1
    a -= avg
    c = np.matmul(a, a.T)
    c *= 1 / fact

    return c.squeeze()


def guided_filter(guide, data, r, eps, device=0):
    # guide: [M, N]
    # data: [M, N]
    guide = torch.tensor(guide).to(device)
    guide = torch.unsqueeze(torch.unsqueeze(guide, 0), 0)
    data = torch.tensor(data).to(device)
    data = torch.unsqueeze(torch.unsqueeze(data, 0), 0)
    win_size = r * 2 + 1

    mean_g = avg_pool2d(guide, (win_size, win_size), (1, 1), (r, r))
    mean_d = avg_pool2d(data, (win_size, win_size), (1, 1), (r, r))
    mean_gg = avg_pool2d(guide ** 2, (win_size, win_size), (1, 1), (r, r))
    mean_gd = avg_pool2d(guide * data, (win_size, win_size), (1, 1), (r, r))

    var_g = mean_gg - mean_g * mean_d
    cov_gd = mean_gd - mean_g * mean_d
    a = cov_gd / (var_g + eps)
    b = mean_d - a * mean_g

    mean_a = avg_pool2d(a, (win_size, win_size), (1, 1), (r, r))
    mean_b = avg_pool2d(b, (win_size, win_size), (1, 1), (r, r))

    out = mean_a * guide + mean_b

    return out


def false_alarm_rate(target, predicted, adaptive=False, show_fig: bool = False):
    """
    Calculate AUC and false alarm auc
    :param target: [n,1]
    :param predicted: [n,1]
    :param show_fig: default false
    :param adaptive:
    :return: PD_PF_AUC, PF_tau_AUC
    """
    if adaptive:
        m = np.median(predicted[np.where(target==1)])
        predicted[np.where(predicted>m)] = m

    target = np.array(target)
    predicted = np.array(predicted)
    if target.shape != predicted.shape:
        assert False, 'Wrong shape!'
    target = (target - target.min()) / (target.max() - target.min())
    predicted = (predicted - predicted.min()) / (predicted.max() - predicted.min())
    anomaly_map = target
    normal_map = 1 - target

    num = 30000
    taus = np.linspace(0, predicted.max(), num=num)
    PF = np.zeros([num, 1])
    PD = np.zeros([num, 1])

    for index in range(num):
        tau = taus[index]
        anomaly_map_1 = np.double(predicted >= tau)
        PF[index] = np.sum(anomaly_map_1 * normal_map) / np.sum(normal_map)
        PD[index] = np.sum(anomaly_map_1 * anomaly_map) / np.sum(anomaly_map)

    if show_fig:
        plt.figure(1)
        plt.plot(PF, PD)
        plt.figure(2)
        plt.plot(taus, PD)
        plt.figure(3)
        plt.plot(taus, PF)
        plt.show()

    PD_PF_auc = np.sum((PF[0:num - 1, :] - PF[1:num, :]) * (PD[1:num] + PD[0:num - 1]) / 2)
    PF_tau_auc = np.trapz(PF.squeeze(), taus.squeeze())
    PD_tau_auc = np.trapz(PD.squeeze(), taus.squeeze())
    SNPR = 10 * np.log10(PD_tau_auc / PF_tau_auc)

    return PD_PF_auc, PF_tau_auc, SNPR


class Mask(object):
    def __init__(self, w=81, h=81, resize=81,sub_w_num=9, sub_h_num=9, dense_rate=0):
        self.w = w
        self.h = h
        self.sub_w_num = sub_w_num
        self.sub_h_num = sub_h_num
        self.target_num_max = sub_w_num*sub_h_num
        self.target_num_range = [1, 32]
        self.dense_rate = dense_rate
        self.resize = resize

    def single_square_shape(self, diameter):
        point_list = []
        move = diameter//2
        for x in range(0, diameter):
            for y in range(0, diameter):
                point_list = point_list + [(x-move, y-move)]
        return point_list

    def judge_adjacent(self, img):
        adj_point_list =[]
        move_list = [(-1,0),(1,0) , (0, -1), (0, 1)]
        m, n = img.shape
        for i in range(m):
            for j in range(n):
                if img[i,j]>0:
                    for move in move_list:
                        if 0 <= i+move[0] <= m-1 and 0 <= j + move[1] <= n-1:
                            if img[i+move[0],j + move[1]] ==0:
                                if (i+move[0],j + move[1]) not in adj_point_list:
                                    adj_point_list = adj_point_list + [(i+move[0],j + move[1])]
        return adj_point_list


    def single_random_shape(self, area):
        img =np.zeros([19,19])
        point_num =1
        img[9,9]=1
        point_list = [(0,0)]
        while point_num<area:
            adj_point_list = self.judge_adjacent(img)
            if len(adj_point_list)>0:
                    for  point in adj_point_list:
                        if random.random() < 0.5 and point_num <area:
                            img[point] = 1
                            point_list = point_list + [(point[0]-9,point[1]-9)]
                            point_num += 1
        return point_list

    def single_mask(self, target_num=None):
        if target_num is None:
            # self.target_num = random.randint(1, self.target_num_max)
            self.target_num = random.randint(self.target_num_range[0], self.target_num_range[1])
        else:
            self.target_num = target_num
        self.dense = True if random.random() < self.dense_rate else False

        pos_list = []

        pos_id_list = list(range(self.sub_w_num * self.sub_h_num))
        random.shuffle(pos_id_list)
        for i in range(self.target_num):
            w_id, h_id = divmod(pos_id_list[i], self.sub_w_num)
            x_pos = random.randint(0, self.w/self.sub_w_num-1) + w_id*self.w/self.sub_w_num
            y_pos = random.randint(0, self.h/self.sub_h_num-1) + h_id*self.h/self.sub_h_num
            pos_list = pos_list + [(x_pos, y_pos)]
        self.pos_list = np.array(pos_list)

        mask_image = np.ones([self.w, self.h])
        max_area = 20
        min_area = 3
        for i in range(self.target_num):
            area = random.randint(min_area, max_area)
            single_target_shape = self.single_random_shape(area)
            single_target_shape = np.array(single_target_shape)
            single_target_shape[:, 0] = single_target_shape[:, 0] + self.pos_list[i, 0]
            single_target_shape[:, 1] = single_target_shape[:, 1] + self.pos_list[i, 1]
            for j in range(single_target_shape.shape[0]):
                if -1 < single_target_shape[j, 0] < self.w and -1 < single_target_shape[j, 1] < self.h:
                    mask_image[single_target_shape[j, 0], single_target_shape[j, 1]] =0
        mask_image = sk_transform.resize(mask_image,(self.resize,self.resize),order=0)
        return mask_image

    def __call__(self, n, target_num=None, dense=None):
        Ms = []
        for i in range(n):
            mask = self.single_mask(target_num=target_num)
            mask=mask.astype("int64")
            Ms.append(mask)
        return  Ms


class Result_Meter(object):
    def __init__(self, keys:[str]):
        self.keys = keys
        self.results = dict()
        for key in keys:
            self.results[key] = []

    def avg(self, key):

        return sum(self.results[key]) / len(self.results[key])


def PCA(data, bands):
    """
    :param data: a 3d or 2d narray
    :param bands: int
    :return: a 3d or 2d narray
    """
    if len(data.shape) == 3:
        size = data.shape
        data = data.reshape([size[0] * size[1], size[2]])
        mu = np.mean(data, axis=0)
        data -= mu
        V = np.cov(data.T)
        eigvec = _pcacov(V)
        PC = np.zeros([size[0] * size[1], bands])
        for i in range(bands):
            PC[:, i] = np.matmul(data, eigvec[:, i])
            PC[:, i] = ((PC[:, i] - np.mean(PC[:, i])) / np.std(PC[:, i]) + 3) * 1000 / 6
            PC[:, i] = np.clip(PC[:, i], 0, 1000)
        data = np.reshape(PC, [size[0], size[1], bands])

        return data

    elif len(data.shape) == 2:
        size = data.shape
        mu = np.mean(data, axis=0)
        data -= mu
        V = np.cov(data.T)
        eigvec = _pcacov(V)
        PC = np.zeros([size[0], bands])
        for i in range(bands):
            PC[:, i] = np.matmul(data, eigvec[:, i])
            PC[:, i] = ((PC[:, i] - np.mean(PC[:, i])) / np.std(PC[:, i]) + 3) * 1000 / 6
            PC[:, i] = np.clip(PC[:, i], 0, 1000)
        data = np.reshape(PC, [size[0], bands])

        return data


def _pcacov(cov):
    _, _, coeff = np.linalg.svd(cov)
    coeff = coeff.T
    size = cov.shape
    maxind = np.argmax(np.abs(coeff), axis=0)
    coeff = coeff.reshape([size[0] * size[1]], order='F')
    aind = np.linspace(0, (size[1] - 1) * size[0], size[0])
    ind = (maxind + aind).astype(np.int16)
    colsign = np.sign(coeff[ind])
    colsign = np.reshape(np.repeat(colsign, size[0]), size, order='F')
    eig = np.reshape(coeff, size, order='F') * colsign

    return eig


def band_selection(data, bands):
    data_n = (data - np.min(data, axis=(0, 1), keepdims=True)) / (np.max(data, axis=(0, 1), keepdims=True) - np.min(data, axis=(0, 1), keepdims=True))
    index = np.argsort(np.std(data_n, axis=(0, 1)))[::-1]
    selection = index[0:bands]

    return data[:, :, selection]


if __name__ == "__main__":
    x = np.random.random([50, 50])
    print(np.linalg.inv(x))

    import tensorflow as tf
    x = tf.convert_to_tensor(x)
    i = tf.linalg.inv(x)
    print(i)

