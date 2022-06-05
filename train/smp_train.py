import cv2
import os
import random
import torch
import copy
import time
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torch.utils import data
import numpy as np
import matplotlib.pyplot as plt
# from segmentation_models_pytorch
import segmentation_models_pytorch as smp

from torch.cuda.amp import autocast, GradScaler

from albumentations.pytorch import ToTensorV2
import albumentations as A
import multiprocessing as mp


def Pa(input, target):
    '''
    :param input: [b,h,w]
    :param target: [b,h,w]
    :return:
    '''
    tmp = input == target
    x = torch.sum(tmp).float()
    y = input.nelement()
    return x, y


rot = torchvision.transforms.functional.rotate


def TTA(model, tensor):
    batch = torch.cat([tensor, tensor.flip([-1]), tensor.flip([-2]), tensor.flip([-1, -2]), rot(tensor, 90).flip([-1]),
                       rot(tensor, 90).flip([-2]), rot(tensor, 90).flip([-1, -2]), rot(tensor, 90)], 0)
    pred = model(batch)
    return pred[:1] + pred[1:2].flip([-1]) + pred[2:3].flip([-2]) + pred[3:4].flip([-1, -2]) + rot(pred[4:5].flip([-1]),
                                                                                                   -90) + rot(
        pred[5:6].flip([-2]), -90) + rot(pred[6:7].flip([-1, -2]), -90) + rot(pred[7:], -90)
class train_dataset(data.Dataset):
    def __init__(self, img_paths, masks, transform):
        self.imgs = img_paths
        self.masks = masks
        self.transforms = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        res = 480
        mask = cv2.resize(cv2.imread(self.masks[index])[:, :, 0], (res, res))
        img = self.transforms(image=img, mask=mask)

        return img['image'], (img['mask'] / 255).long()  # , label

    def __len__(self):
        return len(self.imgs)


class test_dataset(data.Dataset):
    def __init__(self, img_paths, masks, transform):
        self.imgs = img_paths
        self.masks = masks
        self.transforms = transform

    def __getitem__(self, index):
        img = self.imgs[index]
        mask = cv2.imread(self.masks[index])[:, :, 0]
        img = self.transforms(image=img)

        return img['image'], torch.from_numpy(mask / 255.0)

    def __len__(self):
        return len(self.imgs)

def analyze_picture(mask):


    return int((mask==0).sum()/100),int((mask==255).sum()/100)

def multi_count(masks):
    pool = mp.Pool()  # default: number of logical cores
    result = []
    for i in range(0,len(masks),12):
        result.extend(pool.map(analyze_picture, (masks[i:i+12])))
    zero =0
    ones = 0
    for j in result:
        zero+=j[0]
        ones+=j[1]
    # print(result)
    print(len(masks),len(result))
    print(zero,ones)



def train(model_name):
    # os.environ['CUDA_VISIBLE_DEVICES']='1'
    begin_time = time.time()
    # config = vars(parse_args())

    def set_seed(seed=1):  # seed的数值可以随意设置，本人不清楚有没有推荐数值
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # 根据文档，torch.manual_seed(seed)应该已经为所有设备设置seed
        # 但是torch.cuda.manual_seed(seed)在没有gpu时也可调用，这样写没什么坏处
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)


    set_seed(seed=2021)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    save_name = model_name

    BATCH_SIZE = 16
    warmup_epochs = 3
    train_epochs = 5
    res = 480
    learning_rate = 1e-3

    epochs = warmup_epochs + train_epochs
    train_imgs = '/home/project/raw_data/round_train/train_fusai/train_origin_image/'
    train_masks = '/home/project/raw_data/round_train/train_fusai/train_mask/'

    train_imgs = '/home/project/raw_data/round_train/train_fusai/train_origin_image/'
    train_masks = 'E:\\PycharmProjects\\hualu\\raw_data\\round_train\\train_fusai\\train_mask'

    names = ['usPfBM6S', 'FwGiQjEU', 'QdOhflr9', 'QglhkKLE', 'cuKPgqyH', 'hrOE82wZ', 'ThGaqWom', 'iw0mEu25', '29aHeorG',
             '6FAmHhvf', '9ikL1xSv', 'fyIrLhaN', '1xtS6MWR', 'mcjyQuiL', 'AXrRnGUq', 'kVo9XSlP', 'UT1ecqGd', 'Ksz1PqZR',
             'Hhvu4NrJ', 'lNk0xinb', 'YMCrHSji', 'BSMupOt3', 'TDV2oJhj', 'hvA9MVNs', 'gKkm1zlU', 'jG3x2opk', 'QaicZFK4',
             'MiuQ6p5h', 'mW9EjxUi', '7uqSUMtr', '3BnagcOo', 'YSdUX83E', 'xMqonDSJ', 'NTR1pra6', 'VlutkNXM', 'fstqMSyT',
             '059MKTyC', '8MDxW6XK', 'JC0uSzPW', 'XK2QGjqr', '7kmNPCQy', 'sQWIbMcV', 'aKogy9nt', 'MDqTSOd0', 'iXbUSVvy',
             '9jgO8sfB', 'Q1j7lckw', 'ugDAWKLF', 'nA8txCQi', 'r6Ghp7kV', 'jdNWUIpo', 'PmN3BTUx', 'j9d3mMEI', 'UmTOqQSB',
             'Z08l1k6m', 'jgOWovYA', 'yeEbP2qY', '3SjQO4v7', 'BEKO5Y6D', 'wBDkLhis', 'sI3d8LPT', 'qsPRcmfY', '7WcM3gYx',
             'wrbI7uqa', 'bfcaRIQN', 'dPXYkJQW', 'RtvgpJIo', 'zof3grGx', 'PxfDTb9M', 'PQqizGkW', 'Y0u64XC1', '2CheQHnO',
             'NHFTI6Do', 'bM0ziKsC', 'qVnvWQ7U', 'eWIJ58G6', '5KrgDPoA', 'zwbtaqJG', '4a89OPCL', 'GlVQt0Ax', 'QBXdP3VE',
             '4TPf9BGr', 'pwYh7kxu', '4cyLPFz6', '0TCUxIWs', 'Q0K3amyr', 'zebNMhHi', 'r6scLKM8', '1QhZ8GJj', '7SI5OQpZ',
             'maYsyTth', 'oUHPJVNk', 'Sy53tMDo', 'onQviYN6', 'BbWLk7HT', 'lcN659m7', 'EUL4NKQg', 'iowmPlJ4', 'T6M2ZkS0',
             'Hd2WeRxU', 'MmjEiZVI', 'SndHiEMz', 'PJAfSoeT', 'z2Pep618', 'WHGXuPxy', 'OH6EoN4e', 'HZAa82W0', 'lp0dgqm8',
             'qCJSMZOg', 'v8qUmy9F', '7prNIVbd', 'eyXTKF1E', 'YCHEWMRv', 'EoQivOsz', 'cePdtm7B', 'vAzBZ8n1', 'Il2qX9mF',
             'tuqmG6g7', 'FCeuzoM4', 'vdEPS8jX', 'UKXPON5V', 'ZgJzG1re', 'K5Srzckh', 'yHs72ZRU', 'e2olWDKI', 'e9rLIXN8',
             'soPzM31y', '9Ui1y6fL', 'fdoKR1iT', 'rVp56mNY', 'hnavNZ5m', 'rYI4aRXA', '4mTFoILz', 'hP0wkR5e', 'm6pgIb4P',
             '2o9sSGKy', 'dq1JEbCz', '6uRe0KY1', 'dnUxgGIo', 'NVZPuRSd', 'TAStCmwz', 'NaOiMATD', '7FbupKhy', 'b7zFexsO',
             'uHFOgxG5', 'dtl3jBcX', 'lioeKxFk', '54DYSGJr', 'riA0dKoG', 'MEPgwIt7', 'O0E1kvjF', 'IXwjcZ85', 'kxMolVpR',
             'Osiz9V62', 'GO8wHp6q', 'U5IYoSj6', 'sl2iqXmH', 'qK3Fip5R', 'yw9WnQuc', 'rVAu3qFT', 'pCJP9xWh', 'iwkzPjXp',
             'T4BpHcxN', 'OK2kPWdU', 'GQIiDbKA', 'qMzj20h1', '8KX4oMNp', 'izld8cbp', 'ovOQSYn0', 'xO2sdlfG', 'd41aYQLG',
             'U4dEi1nm', 'cb8RPEHW', 'JoQncCT7', 'hwlbHGit', '63dGyUQ2', 'udcFwWmb', 'yKYIDj4P', 'G74kd0iv', '2sAcKoBq',
             'k50qmyCN', 'OngUAEPY', '3BrYal27', '9hjgFL7N', 'PKrCD9cQ', 'qEY4loTx', 'la4Y52kj', 'bePkmB7H', 'l1qp6YhH',
             '7FKPvOTx', 'CAexTB2b', 'wsAaevcX', 'JFTUZES5', '1KxMJtsX', 'hHYDucpU', 'BJx6oznS', 'Ns8qd2DE', 'uJcAmSUz',
             'u12Co9As', 'C6PlbdAv', 'Pmc8srpf', 'MzyvKWmd', 'zVflkHTM', 'opv2BZ4m', '5KQHoRIb', 'yoc1F3fw', 'GX5Bqr0h',
             'JDiBUIrH', '1H6yktpQ', 'TK8F1OUA', 'Sc3HyWVm', 'w3WUbFe0', 'iK1CRGJa', 'Ju0q4WKh', '9ok2RdMt', 'uUoV1Lbt',
             'KaAjNrHU', 'kE6bFVIS', 'tfIKDO9W', 'EquwMAb4', 'fv9TUgIM', '9xLvEUl5', 'eb9wnHSI', 'gay85QcT', 'kgs4y0Lu',
             '3Mlq6ztN', 'bA0hoRyt', 'YcLJefOu', 'kaMr8SBG', '91I7HWcY', 'sFrn1ILe', 'DtfP4W3C', 'EbnZ9V0P', 'UxqBayXZ',
             'CsmKOeEx', 'f3pMioqv', '1le9hdmj', 'zQXLAU4Y', 'C3mPW8Od', 'qhLWIryf', '5yWVbLuK', 'kZNSH64b', 'nDxCMIF0',
             '73ZzUgRb', '8avtc4Pl', 'xELineGq', '95K6Bqcd', '2aTPNdFZ', 'kaAwqFz6', 'AKf3xCZr', 'dbnMtZEG', 'e2PM7XHv',
             'GAfExQ7d', '54yf0EFY', 'KNq8U9DY', 'uWxAyEh4', 'BZKLVuUa', 'VqtsgXyn', '7PoAUGWb', 'RnJhG2eD', 'uGj1YCaW',
             '5o7YTHe2', '87Tmfkj4', 'wqvK1cWd', '0b3AOarP', 'yvlhiwO6', '89Arq2GP', 'yHxnq0ph', '2Fw5xc0f', 'IZ8BLEKC',
             'CvGSdLgR', 'NHY185MK', 'qf0K5dM1', 'gOHWvE5I', '6JWNX0Gy', 'WatDRmH4', 'm60IciAH', 'E4A9hXPJ', '9OofzCJX',
             'in6wVoaz', 'Py1kpW3J', '1VD2COZA', 'CYt1maSz', '29dIk4Gq', 'xoL3a7fm', '2NFeZ8Qc', 'G9Krp7cq', 'qXRi4Woa',
             'Q6ZERkgo', 'sfXPJ4W5', 'KQ658fhP', 'W1P3z5xj', 'd2g0tu3i', '7LeIqgn6', 'NOSHuae8', '8A3GtFQD', 'jEFUuT0r',
             'H96buiSz', 'B7pzuO2S', 'RWzxFXom', 'PuOEaWt6', 'KY9ZrEbA', 'FOC17djP', 'KHOaypBr', 'RypJ97sv', 'U5MEKvhD',
             'cGhn26uk', 'i9Gl2c8j', 'aUn8LpKF', 'QOMN1ueE', 'JwQk1R3q', '79XdLIkz', 'TUmg98HQ', '70l6KbUq', 'kKMC8uLa',
             'jzXilLgc', 'oi1YuGj6', 'qoY9dJis', 'BlnicK26', 'JYhiSlwc', 'bVO9tdvq', 'KkFvQdPz', 'MYRXGNlx', 'Q1AcMPzK',
             '5Fkg96KS', 'BfMDu1WF', 'uRLYaUez', 'Hsekya2D', 'meMXKnO4', '7YJFevgw', 'fPRbuJDQ', 'kD2jgY05', 'yEpAC7sG',
             'YfR5VdbL', 'uzXlKGpW', 'mv4INasx', '9G42rkTq', 'tAW4D5nu', 'vkJMymHZ', 'hRfvPbgu', 'z1vrPm8M', 'yExMID9L',
             '0CbkcpSG', 'A8ganmkR', 'YitkHqZj', 'JinmFYSv', 'UzPr7Yli', 'H1zmgXJB', '4ECm20rX', 'elxvMJ2s', '25qvleKF',
             'YVr39Ug2', 'vzBVAOdD', 'afvxMABW', 'CkpYsEcz', '6n1aRFQC', 'l4Ix5WLy', 'JHpOTnla', 'qJvr7igj', '0Idj378y',
             'Qv0KxeZl', 'JIFL9KWt', 'C7Hg5hZl', '0uAqnERL', '2SBRtTWJ', 'UZoHDMwN', 'WO2eFaog', '1MTplHan', 'Ny4wYRJA',
             'iBaH8vMU', 'zgXx5Ma0', 'yHtcKJjQ', 'W2clX7So', 'c0oBJsnt', 'aPzJmxKo', 'sGqe732T', 'OB81wqrW', 'eS23L9Zj',
             'ByAu9mJT', 'CN1HA5TD', 'AG3V4d9p', 'lXP8GZJr', 'u4TIiaLb', 'IV2LsqBn', '9GYdCvsU', 'mbD2xH8O', '7YbqymnZ',
             '98tCK4vH', 'Dkq5LBHG', 'jinCD8hw', 'E79aOWqt', 'ljLWOPYS', 'LaCujgHU', 'GntJIsCo', '8WiYHLkZ', 'm3j7JNOZ',
             'BbpqITMh', '8QILhraY', 'uRTlO20E', 'zWYaXDTc', '8FEpUihe', '0Lnz8Fsj', 'Zjpvu6si', 'i23CpDtR', 'Gzpsceon',
             'MQroCeJx', 'Y9uq6noX', 'LXkgzneO', 'JgMPOdDW', '8nNmxkHg', 'F2srfzXi', 'wzrdAoUL', 'F2iaew8o', 'qBKfprVF',
             'qLtyBoQX', 'A5ovXSWa', 'cWl50xG2', '5cdAeqzC', 'ko7ZKEhW', 'CVwBKRPn', 'YiPzH3Ev', 'Y1jSKLPZ', 'dPSvu0Uh',
             'HuqDg8z4', '5NBTRGXU', 'VMsEWoBX', 'VtUAGkjZ', 'pinbkwTv', 'lEdV9QYv', 'rk8ovCps', '2pfVBRy7', 'MVGdYeAq',
             '9dY14pig', 'qmpSD0I9', 'zPUgYSvZ', 'uYHP48SV', 'FPJe0RQn', '3wx65KCg', '5suTlbrQ', 'cLECkf9Q', 'ZPkJaoMR',
             '4tJcCaGH', 'z8WUvLQb', 'Ig6xU7dr', 'wuLrRkIC', 'hixyt3Yz', 'WBRExqLs', 'uwSel1Vq', 'JOWDRbrE', 'NwLljpyZ',
             'wxWYXSVG', 'EKTj6epZ', 'hyS3LItA', 'hnfZevUu', '7eIvhzLF', 'sGlctURb', 'rvBdJmNj', 'LZSkJVov', '6Pmn4rE8',
             'J0XImtUL', 'lfIKVUq3', 'yaJGw7gR', 'PzYJHnW5', '1pP6tAaQ', 'lw2JTGBh', 'EsZuwCr2', 'HX1eYUNM', 'bylkdF0s',
             'HPux1MLd', '9dMkQgtv', 'aLwfpdNM', 'vH1Iw934', '3fDoWQOU', 'tamOcxI5', '4TG3j2tQ', '5VWUDNv9', 'r2ilIHMN',
             'dFyAoM7D', 'scrZz1nN', 'RK9UBv60', 'v0dQmgqc', 'x6i8SKWF', 'aAD3fLBu', 'ByxsavIF', 'ka2Dz7Zv', 'ITq5lFXf',
             'jEWM2xT3', 'PfSqceF0', 'NX79tHkg', 'E9hXWriL', 'IkSOAJlD', 'jxYPocsA', 'DNGubF5s', 'b7LrpzVy', 'ydmwxORl',
             'h8KRXp6V', '63AbH9vO', 'clwoGpTS', 'bqAYRGpn', 'pVwUtWk8', '3BzAjMl1', 'neS8HiJN', 'Vlwb9ZC2', 'g8Ld1pat',
             'X5ucqY7Z', 'KSAkGUfM', 'poZR2TLw', 'djXBqI2K', 'ca40nWbO', 'l9QiqzUr', '7IGTfd5K', '5kmFxBy1', 'AS6et8Iz',
             'RICJigcm', 'gqiouahs', 'oOjYpNGf', '4bAX3JeW', 'QFhZbrL0', 'oPy7m2jV', '2ZyNFiOE', 'PUHWxDTS', 'W35hX48d',
             'lohJNI0E', '6Mnmu5dA', 'Ov6KrYUj', 'tKXY0G5c', '6GNnyChs', 'bIznGY32', 'ESRyx58p', 'dSUxyZ1T', 'qFck94aQ',
             'RsVDBpEr', 'dqJZA64b', 'zC0NLlnj', '6XzdY7SR', '0VBxIszQ', 't4dV2JLm', 'LFnRzvkN', 'OSdflL3c', '3PKHMqIt',
             'Whqy2RDn', 'fXzVI0Wc', 'U92aB0C1', 'c5CrnDwR', 'S2dT5ryG', 'do1Ziqc0', 'UA9DsBL4', 'hTr0fd2A', 'mBV7h3at',
             'dYoTcD6r', '4Tuiv2aK', '5JyKpu0t', 'H1xsgWBA', 'cEdPUjMQ', 'w10DJiKu', 'wCpJLzvA', 'RmoCsawz', 'bQUHAwfG',
             '6xtAM2GB', 'jyPRbIHK', 'kQUKhpxr', 'TWRysuK3', '8w9OIRvN', 'AhrReHgn', 'C3wrqIfz', 'j7gfZs5n', 'Nxkb1Y5L',
             'ETu7il1r', 'dD2aKxvk', '53ca0gGz', 'nst8Ofr2', 'HEw2catF', 'HIWtL187', 'KoZQndPc', 'u7jNtwbk', 'jDQPdFgO',
             'pL4PsfiO', 'YtHN7VWG', '4N2CKIWz', 'X2HucPUM', 'ojKvdaRM', 'evw2xFQ9', 'Lf9hNIW6', 'EjBFZRqy', '4RxyUhmO',
             'LKBqCl2z', '3DptMv6c', 'YfsSV73t', 'w54AZdkP', '8xIRYmfJ', '6AEUIRw5', 'YDm5sHJu', '5kcxCHbs', 'S1qp96tD',
             'iDNFpM7W', 'NpTrEKYF', 'stGZXIMQ', 'tmAePfwn', 'YwAj1PZc', '9dnysMCk', '8npBuat5', 'IXMa8efW', 'UFxSlZ6W',
             '70T3WOEj', 'TG0OHdcA', '3B64WqXf', 'raX0l2FB', 'OfBx25MX', 'e0RMoNUz', 'kNnHPALJ', 'OjGx4QUZ', 'UPhRrQz6',
             '7egNbTJ2', 'Cw1JkDsX', 'MubIlUQH', 'KUBfJH4Q', 'cuRHblUi', 'tpqGFQAd', 'HdesJVRI', 'CjzYFi9D', '2zuHCcXA',
             '94ZerBC0', 'aL9nJPeV', 'VvZtEpaH', 'IpiFG4TX', 'zm3dJaxw', 'RBxwvJYV', 'NFVwkbep', 'R4hYF8Xn', 'EJKR9bte',
             'C5mIaHFf', 'xWRa6uvf', 'C8icm2qu', 'LN6sDvmr', 'EvkOAbD9', 'rAl6sfTc', 'SWQevkap', 'JLP7CsKE', 's9eZtWGX',
             'jl5v42Ve', 'lknQbKuv', 'TBKO3JaN', 'o5hiqtcW', '3AwdHKoF', 'lHoOcQYp', 'w1SJrA0t', 'x3R0pi4I', 'JFK6MWsq',
             '09aCsevb', 'NkXrC3I6', 'YTxQRLdg', 'YyiuHrOc', 'ihKHUJ3t', '4gGBuvVh', '6iBuvXgk', 'hGbWCJvk', 'pNzwAKP3',
             'J6zHrTWO', '97IknJdO', 'XouDHhFs', 'eQSVZPBG', 'BASzMQYn', 'VCXSDsW4', 'GHyjMzaC', 'we0ErY5X', 'gc2f6mWr',
             '0GyImbKf', 'hi39QB8s', 'sHuaXh6J', 'rS1tAWxv', 'tOujU36F', '9n27JwXp', 'onLR1eXB', 'mo4iWKxk', '6d0wTqXk',
             'Ob7A2Lki', '7H4wxBt5', 'FBmtXfxI', 'oHwAKI32', '7yuclbd5', 'FXOtSEZ4', 'vRIwkq5U', 'ryEQuGjN', '9btYPalj',
             'FlYBObRo', 'MdOuVY2N', 'q7se10aj', 'pKFx4dH3', 'ecsmR1D6', 'TL9x2mVU', 'V9He03Sw', '074wuchG', 'sKI3Oom8',
             'K2tkyPRe', '6uHZlTEz', 'FUv46d9s', 'xyLRrXeI', '8muKFIq4', 'KQRfqaZM', 'Ut38edQN', 'EOBHaxzY', 'F4io9Oue',
             'MBQvs35C', '9VtAixcB', 'OCmkJBT9', '3GXQaiLT', 't4pLAMq8', 'C6TOQzu0', 'YDFMcjgo', 'VA8nXF1o', 'hKRti1WA',
             'dbkBzuxq', 'tOeIjBL9', 'v08eCmZw', 'QBysn2Fh', 'znmfxiVI', 'xJGBbU56', 'tHN2YSBr', '7tZLHrO9', 'sjR21WVl',
             'CXPMk8Vq', 'NVotJPA5', 'ZFHtfOJR', 'a2ULGoXf', 'bjBYeayd', 'vBAdtMFV', 'T297PCbu', 'LmRI52oA', 'tCpIOc64',
             'pCJLYxdg', 'm1VUhvtI', 'v48GoQXO', 'U5lDJ6jh', 'fnUTsrFR', 'luRV6YZg', 'nkrR1asW', 'dJ96KyDs', 'm9EgU1VP',
             'ucgytqfj', 'e3K8fYSE', '10jv3tBb', 'E9IqRyaJ', 'Z6JXbrQi', 'UhCJoFms', 's3CIj2VA', 'FHJdCXgA', 'uz3glk7j',
             'BZ5fc4SP', 'VJpW6rOq', 'OescqSDj', 'gy4Fc8YB', 'pHG3wizV', 'nZKFQ4oB', 'jXPQLp6b', 'vaf83XBC', 'DY5cmSQp',
             'vthXDlHP', '2wntzkeO', 'GcZzTaxf', 'MkT8szhw', 'W42b6H1c', '1xT3oW4D', 'qPYRMCnH', 'wuR8aAfG', 'ngzdUNVA',
             'k3YnRQab', 'WSgi7sKr', '2LHAEhsl', 'C1xzLm3t', '5Efm2art', 'TzhCxVWY', 'DkU5RNfo', 't68Vbiph', '6CaRhkIG',
             'fpmZN2Js', '2vNy7gAn', 'MOf19W4l', 'K9aQ70Eh', 'uSpt40wh', 'NkoiPmGC', 'mTI0DeuQ', 'ypau0ikN', 'u2Q5nKpI',
             'CGXjDFpV', 'KwDsI2jz', 'rU9FZ6AK', '7oxALzFq', 'x4ZfdOFh', 'iVbNjQnZ', 'xngofBOy', 'IX8ASg5x', 'ZcXutKsr',
             '8MhbPA3a', 'C5X4Gzq2', 'Pw5aXbvf', 't60nlU34', 'ldpKokIn', 'QqLviFkN', 'uLrOavgl', 'GtmMB1oR', 'iEohzKls',
             'TzbOhWEn', 'XngD3Zr0', 'dTurDk7l', '9amRJ6nI', 'dWcgi5Sh', 'Ytwd3ucA', 'MBtIYEP4', 'njWRQk31', 'GTNtFfE1',
             'IGS7ZR8V', 'tmAiqHGx', 'VF3q9SWz', 'ICGt3654', '6RMLqCmw', 'EaHJgZcG', 'rZjbWh1L', '2Jn5rqjy', 'QMrRFCeJ',
             'WATDH70N', 'NQWKn8Tq', 'HoAc8tys', 'ZsXFAhPO', 'SR2lACgf', 'H4dIKahc', '50xlyrvG', '4Z9wfDxi', 'dv4ja7ie',
             'F0esWk7Y', 'HeOiA783', 'vWu305jM', 'JaUgbVDw', 'BOQ1RoSZ', 'StIxTfkO', 'Jckh5grj', 'SyzZ8jku', 'hWmwARdE',
             '6U2vw47G', 'Mznc9xlw', 'D2Saeldi', 'eE46HDgQ', 'TcUK2wf5', 'TdXMgpir', 'iDx96ugH', 'RxFjJ5wA', 'vD2zL6ud',
             'xc1i35rD', 'AyVrgi93', '0TYXFhMN', 'SFw7Au9b', 'naRKHP6o', '3vHM9nIf', 'H9XaGM84', 'qCMH4kyx', 'F14EPqTX',
             '1Ksgvh7F', 'LATQ28NC', 'cFVoRnwb', 'iZT2c1an', '6V8HeM90', 'N4I93WpF', 'bWLSqjIk', 'pMY6mwxg', 'ueFgEUT0',
             'wixzEDC3', 'QrPpyJHD', 'obFvts32', 'cGs8THD1', 'wxTypzrh', 'oFmcsHJ4', 'wuDmeQ70', 'vENhCnL7', '41LvbGmV',
             'IN9se34j', 'g4dZp3x5', 'hvFeaGmo', 'DKRqSsFl', 'A1GtryNX', 'PkQw1rKF', 'h4stPZJw', 'XlpdzT1q', 'rzTpnw9i',
             'Tam2sbgJ', 'LeKNStTF', 'PkuBJwRD', 'MHT4CJVs', 'uaKV23fB', 'YFy7pumr', 'zi7jnkuJ', 'q7MsuSbz', 'IQhOWT1P',
             'pZxjk9UB', 'sbdZ4OcX', 'ZnNDRtrM', 'KfLCPwjh', 'yJM4RHgp', 'mnqGTBbZ', 'Pa0oSJec', 'bZNWrnCT', '8IMytJlr',
             'wGo6NIbZ', 'AMGUVI0n', '1DAqgTLH', 'TA9McdOo', '5nyaBjXU', '2u3mJXwE', 'qCmRXFrl', 'Gw9LJIPU', 'jxvkRqcd',
             'v6cY2VnR', 'ZsByMgi5', '3xCj6nvW', '16u2UXxk', 'nzT0Iyex', 'VTXQm526', 'WVfre8dJ', 'zDeNLcKH', 'g6IMq5ae',
             'KmX7ucPI', 'UCpm6Wsd', 'LnwFej2i', 'ZIlLuUCM', 'YPZUcthf', 'Jh2HC5QD', 'sUQ3bDjI', 'F0HnySYU', 'eq0gA6Et',
             'N1VjQAhK', 'gczftVnY', 'TcFLtM0J', 'mFldqrik', 'LwBjU4Cy', 'Ak7ih2wc', 'q3sFykCP', 'O4ndcWlh', 'kFBLo849',
             '7aRDbHmh', '2TOQ9JvU', '8HJYxqAu', 'CsRQNIbl', 'f789tjoD', 'WTNBd2ZF', 'MO7T9nDv', 'IDuUNj4d', '9ejBO60R',
             'rfv7GyIw', 'oCnEkJ5S', 'HxZSb495', '9E6Lfgrx', 'qesbGLm2', 'ZWqhtLra', 'pUQWu37v', 'uXMpzwAH', 'UzqGr4v0',
             'oTfqVSzm', '24l5Wp1D', 'bZKBd9gW', 'oiQzDP0g', 'bAKhokay', 'YDN48G9o', 'sXEyg9aD', 'IHj2QhRb', 'wftMiDgI',
             'pafYjZoP', 'rBs0tvcI', '1VqtfQuY', 'rEDZvO8L', 'gdpCIGqn', 'HY5pGv1d', 'c2InWfGH', 'gl09ejxf', 'xFBapelO',
             'FdimHsOl', 'REn9iaCj', 'x1jyFrEl', 'cimoSluJ', 'jCb57nK1', 'KpOvA82o', 'rOnAsDfR', 'a7eoqANS', 'OdY9xJFB',
             'B5C9NZIu', 'MhqvQiIA', 'G9cwzpPh', 'POLfH8FM', 'c0kvH7RS', 'dH9MUWpy', '6kahdXgM', 'VnWsBcHI', 'GEnoFfdJ',
             'T0aSFw7o', 'dBNuWGR9', 'ROJKAgWv', 'Y5OJ4IhV', 'r15nYQcM', '3RAGlgn6', 'Gyj2ZJRq', 'bxFoqrlO', '2QSa7e8Y',
             'eqNf1u5E', 'ykGxUiPK', '6AjItgQV', 'rLa7DjQn', '37WkFbd0', 'HyWaOzs4', 'PCdq2MrL', 'I3Zh1BHC', 'nWywRZcB',
             'Ybi3O1nv', 'tUql167R', 'r2yH9TZ1', 'w2k7CJcQ', 'kYBNxewz', '1TZl0auR', 'YEaKoFRt', 'WbRsFdCf', 'Zp7tmE9N',
             '7ge2vYSu', 'uhZveEmI', 'dZJz71wx', 'm5dc1ab3', 'F08bzLBf', 'bgh5Vc7d', 'nXpNCQxA', 'ZjM4fc1O', 'ah8yO2Gr',
             'uWkQcBrw', '6CxSdTQi', 'wXM4EAK8', 'Gah4gn5q', 'LT5guxhw', 'WyZN4HIm', 'Wrjgk385', 'XWkYd42f', '0A3GfvlK',
             'PS1BJ9uc', 'WiNgfybs', 'pac0fByr', 'Gy3hj15E', 'ckbPtzqu', 'pE2SIgvy', 'Rk9faxUG', 'uGLZiMKv', 'hiXpTAtW',
             'E3aWNTL9', 'DrxscV2d', 'iTeZQBF9', 'QPXWsNMI', 'Htg4LuXK', 'qzDFN8u9', 'AK65xDCz', 'OtXFgRQp', 'x7LVKQzq',
             'NuhODbIS', 'ZVTdqhrt', '1Uulk7tx', 'eWBtvinQ', 'LpU67vcJ', 'YoIVjZBx', 'JhRXWISf', 'um8OEdkv', 'e90lXR4s',
             'yEDc7V9k', 'xD0rtJPU', 'k4NWR65v', 'Stv19PHM', 'F7yHzkC6', 'UqYEAPcB', '8TgjdOHA', 'eCAOpLxD', 'HIJFTq7L',
             'jOk8aups', '1F63zJQl', 'Ezja5roM', 'fEF1kL45', 'uNbydKEU', '1H7eC3ng', 'AmcI6NUG', 'gflOJ52D', 'tvEbAUq5',
             '5pSHvcXu', '85OeLRYb', 'NVkdjGhI', 'MGtI40Ww', '2cf8ThCi', 'hZkLNsIu', 'fFWwn3RQ', '1Ag5yWMs', 'pG8f1gCr',
             'DcRhq4Fl', 'iCSsF4Jx', 'gAON8uUt', 'jr6bNPKx', 'KQMC8s41', 'YT4yOGvf', 'pwJ4HQ6U', '7NXme6fV', 'Zobkj5im',
             '9K8j4x1N', 'vHa0zAUV', 'Ikzn3JoK', 'P8SoKvmH', 'keF08QEx', 'S5DN1isu', 'sY6i45oX', '7jUF5nTL', 'pg53tcI2',
             'PbDvFfUY', 'clW7ahNg', 'dYprHhMR', 'WzRhrdfE', 'ml09YOgV', 'Ig7UTVco', 'eO0m5dc2', 'CLqoIg2G', '0khzUMiH',
             'OYEh2dvW', 'Nl7TSqnk', 'hgrTVecl', 'sO23bV5q', 'FIKBd8hz', 'WzqGhF9e', 'txzjbk9U', 'o8SPbFfW', '8LVdc16R',
             'TJp35DCu', '29a0kCto', 'xlaSHK23', 'wPzvGFEq', 'GYrQ6SZ1', 'iX0jxo4g', 'ygdODA3x', 'K3A7tO18', 'k3bHzUM7',
             'Lcy5KtiA', 'jnDXxewk', 'N2c79ZQA', 'yrsv6qDF', 'gYmTeo7c', 'eH6nZCw8', 'QDujVesw', 'iauAHScl', 'ARtsdzpM',
             '3UIGiJtY', 'OD503KRT', '8PgetqnY', 'OFTUrl8q', 'r5UIkvgs', 'oYahg3im', '6Cv4851V', 'wNSUhtW9', 'RJ0nbdUD',
             'L6USlept', 'JfrkQGwT', 'S8wUgt7B', 'E5T0b3VS', 'ezwh6Vod', '9biSUwZ2', 'JfpuMcFk', 'nd8qCAWX', 'LzojpX2A',
             'E3DFWcS2', '9GBCDFbQ', 'hOIWZrlC', 'jtCEdTNB', 'Az9BH50X', 'QbW30kAC', 'vaetETXk', 'NCZXGBvz', 'nwYU5rzs',
             'fipU1Qel', '4jlCaSfr', 'oSz5MUTc', 'bZWwrs4e', 'PK20571h', 'NRBCkMd0', 'JI9qe63W', 'qW7Swx8H', 'yjeOKM5b',
             'Cx1A5UB2', 'fwC09aNR', 'zJu3mFVI', 'oIirAPXy', 'BcJGPCyM', 'nipTQGuK', '8bvqa4JW', 'nBLlb0st', 'UsriKW6h',
             'oBe9O7mL', '2vsIwPOy', 'bLYkaBqc', '7bIO6r4C', 'EjpzeiBM', 'Qkm0TIuE', 'a8TwrU5s', 'XcvDnrfC', 'ZyBdfjau',
             'alkF5ryN', 'DZ7v6oeE', 'xDhVYPkU', 'T0vI63j4', 'bOM6Djtp', 'Ho8fJ3vh', 'kHPdrwnV', '3FLmUPeK', 'UzV4kC5R',
             'vWnPZMfC', '0N67Vzeg', 'u2ihDk8s', 'UzISuAFG', 'Jt9vEzgd', 'eCTzv2xR', 'cHaMthZU', 'ajK9qhS8', 'AVP1awID',
             'JTaXsB6H', 'QxIJGlf9', 'b6KAZauI', 'oHswFzrg', 'Di8NPyOW', '8jpGFXbn', '1WXj5hYO', 'z3YrujPG', 'CTetsZA7',
             'LhRwUz2I', 'nARuxPDK', 'nQvjEYCX', '9cDpzsmI', '6K1F5fjX', 'gQnk3zmD', '9J1cmPx4', 'WAlIcr0w', 'FzEsm1UC',
             'xOVJFzWS', 'vTR6aXNe', 'nL8tSqf2', 'xVopRNGs', 'PZpB5eOs', 'zOurcMd9', '0h5gXTcq', 'q7cKAERJ', 'CLUlQjhq',
             'xIDd9mOC', 'rCX6Nq08', 'umJxLKcT', 'XENmqwji', 'mgpB3Rso', 'JK8BhLdr', 'Ghmd9iHw', 'Jk6bSNC2', 'ikjzR08A',
             'K1njSMme', 'msbSVKJa', 'rmqAGIgD', 'OeYs619J', 'YXdLx7Sm', 'yVlx9HgF', 'FJMLpCdK', '4pIS7hCk', 'fYijC0JN',
             'fGcNZedY', 'Y7CiqwHp', '9r6FXIR5', 'kWEl5DiJ', 'anjxpeol', 'dAnQEokz', 'iPSNMRKt', 'JaqT2ftC', 'bxFBnYAv',
             's2ru1oJd', '1qak49cj', 'lyJHSYNu', '32cSUlRo', 'gJKlLUq3', 'KscyJGj2', 'FMxhkKSm', 'H8QGsjId', 'OhNTCZVU',
             'gTpeE20r', 'EtjMvfap', 'sOpZtMRu', 's5alZRGj', '0cP5U4Tr', 'SfzZIwKT', 'q3mcQbBR', '9LTNrvpa', 'k5BOGpMH',
             '0L8GQKsT', 'JzNCO8A0', 'EBPUVRs8', 'DdcwS4WX', 'CctGuH2j', 'Srziy3UK', 'xqYACtPr', 'zaUqZGWh', 'GUcesyAV',
             'q2LI5vd9', 'Q4BVFaws', '8zw3kWbM', 'zxCZapB3', 'SAVnCzrL', '9gxOqBAs', 'uwDE6Ger', 'TYjtUw7V', 'wh6yCkGO',
             'TSZh3rot', 'pX510SuP', 'PtvA6kLI', 't7FyWlMf', 'FHjVMLuT', 'h6OgeYET', 'h81uWzFO', 'kNqQYxCT', 'ZGTEhe7W',
             '6qNXJ78E', 'wbN5ERCm', 'XD54dbFH', '5Ty9APBa', 'CBfyZJXN', 'oCz0Ghle', 'zXy0f3cE', 'Gt9XcoY6', 'SmDV9HJR',
             'wxyiAkHJ', 'JKGzFpiA', 'eNlwGc1n', 'ItzHGdQJ', 'aruwVMyx', 'IqKAgtcO', 'ZoO9AwyH', 'QufSaJe0', 'Vaeg6t5x',
             'n3zsudfT', 'dhNxpAwj', 'UTVRdbF5', 'mydG3xh8', 'ovg59zFn', 'vp8Une6u', 'bash7fIt', 'UetjnSPO', 'C6YZUTpK',
             'ZxWqzLc0', 'B3OeFpJq', 'kWGZxpts', 'Tj9wP1C8', 'IhXGKxw3', '6e2bVUGM', 'VFlvjPxf', '19drRgZI', 'byzR6Jg9',
             'x1VgB4R8', 'dfPjOByY', 'a0J5sH4v', 'jtL9uDPf', 'gn6F98fZ', 'JGLd8awN', 'XB2caOZg', 'fvSdZmhI', 'Vk7L3ioC',
             'VMvrkT9E', '4GaxeHtu', 'lxAZRbi9', 'nNCzoW1Y', 'fAIiY6Tq', 'cuITGktJ', 'zwqEyV17', 'Wy4j3OS7', 'C3qLgW0y',
             'EIfirbDy', 'wyRmsjIl', 'rFEzHn7W', 'g8f1rtV4', 'cQtdPHs9', 'aPofQ4g9', 'YW2Qgfem', 'frv2WZ9E', 'LyKwCFfh',
             'JC7wg59K', 'D7h6fNEB', 'ehiZa4Ap', '8wFYvDPZ', 'DLJiaA8n', '5aS2zGJY', '9kx4rzKG', 'vLNcTbQJ', 'eLIQKJqf',
             '8rSBWR24', '5NoA6Y4P', '3HyMEcpO', 'IFbAqyP0', 'TVY1FyS0', 'qIAHGOJy', 'wYLfHqyn', 'iH9yRCkm', 'ZGRLBPVU',
             'hjUl0Hos', 'fSk6YOgj', 'Xp3SgHvK', 'AtCUgWwp', 'Zm5wP8aK', 'QSp7MtyH', 'MOHGNc1l', 'GVg54Ywu', 'HfJDLz9p',
             'xQsey5Lp', '2g4Viysz', 'hVCxYpqL', 'HWgxPG9B', 'OUkbRjQ6', 'dDRHGwvA', 'xOhkRKPs', 'boVnkGhi', 'YCXMfP4u',
             'YXG6RFQu', 'sn0mux3g', 'PGAmn1RC', 'fkjhm3UD', 'eF1Lg75q', 'wATNXhZr', 'ZhYNAplm', '0Qq2OJEX', 'pRJrgAsE',
             '4KdZhiso', 'yQhwYgIn', 'HQK5yhTx', 'AjJYcmyi', 'SifLuJgt', 'O8strIX4', 'o7I5gJEd', 'SEbNW7xL', '0WJ4xMKw',
             'mGfS9g3k', 'lWTAk8cY', 'DaMiuX0R', 'dGbiJgkD', 'QXNIcU4G', 'zluJbHYB', 'SH6hua1t', 'TzmCH35u', 'o0HK8msf',
             'BgZU2w0f', '409Cfkyj', 'cw3JmoZD', 'Vrwu7U0s', '8WV5yPnC', 'Ks2NVi6F', 'IE6lRG1i', 't53hH4LY', 'DfmTiHXl',
             '7RqCZJkv', 'mudeEhsF', '3nTdlXuB', 'ydarUDYB', 'PrMHieO7', 'wLCfyv4V', 'WACoLKxe', '4Yjbutlq', 'PwixQmJG',
             'uE7CAkzW', 'vsyC5rXq', 'tlCdEPyV', 'DBAx7ySf', 'aNXWCy9i', 'lreovE5g', '0mECizoY', 'w9EXlhxs', 'hnWYDU30',
             'v2QYDaGW', 'Q0hle6mR', 'dIKBslJx', '5dFQTRMm', 'fX5Di3kF', 'h9ofGROx', 'mdqnAGoK', '8LhyDBpI', 'T90HeRKu',
             'OuDEP4Ky', 'EvQkKrCc', 'YMKLeRjF', 'jODsGlWX', 'sDS8uofE', 'Kv5B9mMz', 'MEgj5nAs', 'UcnL9H5T', 'ik1oQ0hc',
             '5VoHrBqi', 'b1TwrpPE', 'j2Srqe8o', 'XQlcP3gB', 'qoryI809', 'FN03tn6h', 'TB5foHCs', 'e6cDlSfY', 'YjcfJ1h3',
             'rWmI37bl', 'RoZFSsTq', '8brzHyKX', 'Gb4Qfa6p', 'KkPZvqCt', 'wlDH9Rc3', 'Bkl70ZQ2', 's38LvqG6', '8DV7wt96',
             'vNIsPWhU', '7dsyak2w', 'zUQAqJhv', 'hR9gbcQn', '9fdWX2tg', 'j1hiBsIE', 'Y9ixdIqF', 'EGuS6R1W', 'l0sHvrqB',
             'IEob9ML1', 'bvWqmpj4', 'z4d7v6mU', 'gMOPC7GU', 'XZqRtdzi', 'NUKuO5Y6', 'EZaQ92rA', 'PgiysjeN', 'yBxFO3vq',
             'lUQwTzjS', 'eKoBdrhN', '6tvlnbd3', 'JxFavKQz', 'fOGDqYA1', 'fSU1jJz5', 'vh82TXV5', 'sem7PCop', '8OVKx0J1',
             'vnC4Yk0m', 'IT3Uxq15', '1dyHZbk2', '1CD73q5z', 'DR8K3BIp', 'Lq9QxaUe', 'NalSR8Eu', 'odhYxlXR', 'AWfyVroT',
             'PODqUYpb', 'CrIR4QF3', 'CmigVMAu', 'cCxNQGvu', 'Tai94Cc1', 'jDrmp6wx', 'zr7Re5hb', 'W6U3uzMX', '5l3XGKO4',
             'QaxRTbmq', 'JEfDSdyt', 'UL0eoClx', 'qYdUOxe4', 'B1qXNCU6', '4vCEZLyR', 'mZ9Detbc', 'IevFnBxh', 'MLuyQFIR',
             'BY7PoXLg', 'UACm0KeS', 'ExXf3OPw', 'HpTl7RiQ', 'Y6BE1dOG', 'xLjsTgSy', 'qxjOW315', 'bo16w7Ux', 'OiIge4kP',
             'zB3ExtZh', 'lRjH4P2Z', '1EBI7fNa', 'wAj5BXMx', 'tBFoDMh9', 'sXqohj8H', 'losrkfUq', 'TRpWmoaI', '76pyeDPt',
             'FzSj5dBQ', 'UWcz7ZHh', 'GpJhNXcK', 'IojkYt72', 'v3xQOJYm', 'cKRkAnJ2', '7TrZEvsI', 'Rsla37HF', 'xkYiSP1t',
             '4NMTG7bB', 'XkJfRu6P', '9QHDeOFY', '7JI8XBtj', 'JZHmGOqN', 'EAIjoK0w', 'IdJx845N', 'QtAlaXUk', '4hU95OQj',
             'SoucFl5T', 'ueQhRWj2', 'ph59rZFY', 'khBylMO4', 'F3XCPrWQ', 'vkSaJ1Oj', 'Pl3a15UG', 'G0uxp5r3', 'cTqzP6e2',
             'QrcYNBj8', 'BdHPQRkl', '9EJI6Ytp', 'OYaqwCPG', 'efAVLpK3', '1s5PbL20', 'pOma39SL', 'yhoRb5fV', '9uX2ISO8',
             'Gbt0I3RY', '1xvecESt', 'X0FQrNYJ', 'bcNArQvC', 'GoAPfgFb', 'G7fA8yvw', 'IrjX83zW', 'QpLn9vhJ', 'n2hk5JKH',
             '26PT0LJo', '26ROCWsv', '6dVtHeQj', 'mfC12Ic0', 'LU2npAHF', 'rJCqkfvc', 'k8jzVPZe', 'aAlzD6vM', 'q7kh9veg',
             'lpoeWm7M', 'rpIqNFLH', 'ov4qx2hX', 'tx10oacV', '76K8aYfD', 'Sh42ge9i', 'PSD56Rbu', 'GhiSo34v', 'wDINCJeT',
             '0jwu9xpV', 'JweT5X9s', 'f3z4gFBN', 'mhA4USZY', 'SRIU8kTa', 'qPcbtaWw', 'x9hjIKqX', 'hodMXW9D', '5e9tlvQz',
             '5BQEJWDI', 'Gbqe09IV', 'o7DP4rpY', 'TO6BJd02', '5Teow8ad', 'uLwehsGz', '9hBNEjPA', 'kHE1PlAa', 'BvcnV1hi',
             'Ey2plcL8', 'fojZwsnK', 'PUAyjOtV', 'Os8FnjZN', 'RSvWb53N', 'aAVSuPJ9', 'er6iaHx7', 'Noy59KPB', 'yV7OXZRd',
             'lOYZkJK7', '9QHfBKz8', 'BzfyPHvF', 'A7JiNfo6', 'UG4tKITO', 'fEjMGw1T', 'uQPjVa5o', 'aimfTCsZ', '8mgAoVGr',
             'OLABdQT7', 'OSPZgcw7', 'Gmq97tUy', '16yFSXtC', 'Nh5ZeVo8', 'JX76HhYA', 'kM0ZE7C4', 'zQPcoF9U', 'e63iuR0y',
             'S319ZnIH', 'xcCpkAqJ', 'DBylYG2d', 'tMyZINfJ', 'jZTFceBu', 'Vzigy0h7', 'yT1ZpLNm', '3eK6WJ5z', 'YnbiPNE6',
             'gONk9MVy', 'h12wIMYQ', 'mRu0OPCn', 'ImWCD0NV', 'HAFwYyVC', 'B3cwa2h1', 'tMJuAnXS', '9ZhsiBWd', 'useK5Q4k',
             'hibCPUtK', 'IClda4io', 'XRr86pDn', 'IRpW1iHm', 'RtCPQ8ds', 'vTZgXeV6', 'D4lsi9u2', 'F68YowSf', 'ouMOa7Ks',
             'nFkwzith', 'V0Z17OCQ', 'BlczhtXw', 'l4Fxvnoy', '1zTnGk3A', '92rzK4xa', 'jGvCezH6', 'ihjBRgnA', 'm4UTAsln',
             'k6nWK7I2', '179QadLz', 'yngMe3Jj', 'HIYPuWxE', 'Ve9UW0zk', 'X31EjJyA', 'kRl7sBKG', '0K96BERQ', 'zejM1ql0',
             'xSIgsj1p', 'IL4c9yTO', 'ckY3nvoI', 'IqckChSV', 'uWVxTQiS', 'Q8u2JAkb', 'UVuqiW1L', 'bPi0eRrT', '6HhQPzp0',
             'dGbLHMgc', 'PT3kuKqR', 'xofi7acq', 'Of9s6KGa', 'IyFveL2w', 'rwlhtfA8', 'ZlT1LfBH', '71mDRo6d', 'RI2VFMbS',
             'ZovcMQjs', 'TNR0dCeK', 'Lszjd6Mr', 'fZTWVdMw', 'Ec4rkJXg', 'AuFJVg0B', 'csW8wTD3', 'x5GjSeol', 'usATmfpn',
             'oCPIyFYu', '4coFEY6N', 'q3bylDRw', '4rjOZ3TW', '60XuHPmj', 'aQc7DpWB', 'REoe0dwV', 'mAPdvM8s', 'Vub3GZeh',
             '5uLkf6oN', 'Tod8tViS', 'uK032QcY', 'zmR4Nq3b', 'k5fVzd0F', 'kIrN8zp1', '1EGqkcPU', 'fvtHGmjn', 'eFWnyfQE',
             '1VJv23Yz', 'VjuMqYyo', 'd7HeNyZU', 'tkFLpdco', 'NrsCgSjB', 'MAxaHCuR', 'Gkb0sYma', 'R0mqT91J', '2bLQGYdv',
             'j6aBT3US', 'MtQpszCh', 'ER06xca2', '8ih0SZGQ', 'hBD2b8gC', 'b8i3IvGp', 'G84LM6sF', 'TP8ErskZ', 'JuetdRUE',
             'pQo9DrSd', '1bB27Zs6', 'yh6SIvLn', '1wJNQK9q', 'nCojm6Ny', '0bLxCiJf', 'h2MgKEdR', 'jdDSUlaV', '60uRBAj2',
             'uvZNtxiU', 'SAWFzl5U', '4iPC13Tr', 'wtQlhfPn', 'jBQ8JOeE', 'JpiXmV6W', 'cMA5yYsW', '52zDK81E', 'l87xSIDF',
             'YQcfGC23', '1nc2kmFS', 'uvJtry5e', 'HuFp0ceA', 'W0PIQanA', 'bqfgCrnZ', 'y9fdilrT', 'SFBG2mZn', '8WFXSTEQ',
             'r2XUIZGf', 'i3fNxMeA', 's4mMjqG1', 'Ulytpha4', 'KpGTIlqd', 'VU3uFX1s', 'aEOdAJjy', 'S8jkumCr', 'frpSwDyk',
             'Jv8PUYmK', 'I4h09OsR', 'SQODZl3J', 'JWzvECDZ', 'nrkBRmsC', 'pkvV5Zit', '3VQNqHAB', 'luZIViQq', 'aN9MC8kV',
             'y9JPiFbn', '6QIceiWT', 'fIM7kGDh', 's2a9Lndw', '0T6jbwPu', 'KGASCp4R', 'VxYOIm0b', 'Wlk6f1Cs', 'q5TkMwGS',
             '10G29Die', 'uS2W7ciX', 'wA0OoKtY', 'DnXAOymj', 'n1dMEH3U', 'Kk5feliY', 'aMJ61qnv', 'fnbqKGVA', 'KJ2d6x1N',
             'RFIHmjYd', 'tpiGw3mj', '05IvTbwA', 'tqLEH2wA', '8RBKQiN7', 'hu8lQ6B4', 'lyHtNPm8', 'TjvpcYgt', 'NWcmbfK1',
             '4ZbF3SgX', 'XPQl5j6O', 'SyE5XOcD', 'qzfIEomy', 'cetHNBdl', 'Lj8VIc5H', '74biBIm3', 'PTCH0wAb', 'O49U2APZ',
             '6RAonVQ3', '5fzt6jxE', 'YOJLXWpi', 'OFVisJ4I', 'mqV1g3GR', 'AFKZrvlt', 'hI1nTRqz', 'sZHwAibt', '6wyJNIfL',
             '5ctO1esx', '6P7vA092', 'auTbJOYc', 'PYI6DBju', 'CpIw7t3o', '9NYPuCyU', '5GBAjQwi', 'O2M5tIN1', 'lbcU12gp',
             'mv7g4n1y', '0ETSH3C5', 'XpF27TWg', 'qG28xtoB', 'OoqjfJ7t', 'P2eT4yt3', 'npPcRCKh', '0b6AQImx', 'aT3mXOxA',
             '1m0FYdM6', 'GxMzm5VJ', 'akoPM0Tx', 'p8la5g6x', 'mSC3aZGO', '1C2usAjn', '8ZwmAIeh', 'ZfykaV7U', 'QMPlgoGB',
             'V3tnZCFW', '0JCKeLdj', '83oYZjCf', 'yI8bHOYN', 'epTNS86j', 'uGyn5aU9', 'wEA2ouNt', 'BfRsSeCU', 'vchpPgZa',
             'OM9paqk6', 'BiRb4pCj', 'mYDN31Z5', '7rOBRAj9', 'cJLiBqDK', '70HMmI2y', '80kLDeHr', 'D8yLo5SF', 'AlyfE4mw',
             'auDmBf3x', 'BiS5AjMx', 'NiB86L5x', '0yoN2Tg8', 'CiBRjIXe', 'rzcf7hpG', 'TZp4V3h7', 'VHb8MW6t', 'di6aVTE1',
             '3aLrRN8h', 'ZYpxS4KL', '837HfgSw', 'kQDApVIq', 'aQg8s9nM', '8IUN5onl', 'iL1ERdw9', 'TV46zQHs', 'xClTQBpm',
             'wFdzpMbJ', 'bs3gvV1P', 'EybTSDJm', 'HITQnPN6', 'mrdvc8lq', '7d2SZaCq', 'b7PKgFnl', 'RkUvyGFn', 'mnPtsoGE',
             'CzwQ5uTE', 'rUNQtlbk', 'Y8hm2zZC', 'qJo0k4xQ', 'yNLZiBfl', 'cm1ubWFS', 'W6C9ga4M', '2IThHsCz', 'rSTyfHGV',
             'PySOpb8B', 'tsFSnVgE', 'qBeOrv68', 'ZpKOat3P', 'PEb7VmhJ', 'XNfyIEjt', 'kTXfRGKn', 'pkcMYBGE', 'Ccu9J6tb',
             'RJeFIbw8', 'PFLNcUa7', 'glrOPd8U', 'jRt2uCOw', 'RB1wxUTZ', 'OvZrXmSk', 'GmHUcTzx', 'kApxU61D', '7tvEC4wu',
             'i9sk8EI4', 'TkKeSAH5', 'S4bw5dx9', 'pHNcBJXC', 'ic4vsgG1', 'JpUt47Gx', 'Kp297oE3', 'cZXhRmDy', 'DitA8XCR',
             'vo27GjTp', 'cruxPEmS', '9gowlpYx', 'ENiUKsg5', 'Wy42z7NC', 'tTGVLXIq', '8sJ6XwGm', 'cuJC5BSl', 'wNaFPkqJ',
             'ncrPYO1a', 'EfpD8zBs', 'GErBT4j8', 'aM5tLpY4', 'ati7w2x3', 'R2ue3fQh', 'jA5l7zqf', 'mQIM4YjU', 'apfnoe4d',
             'qBlezVOK', 'se2UJ31y', 'aLcjfCq9', 'iqWwGyUJ', 'T6OrFmje', 'K03m94Op', 'SJ1G0AhU', 'hyVsZmxf', 'F0aYwSDb',
             'Lq7y1rKC', 'RCsXAvSI', 'Y2LR8jNX', 'DlTvGu32', '7gbShW5Y', 'BHdSg9aj', 'ra6z0wvQ', 'K3AvX1CG', 'fELZBq8v',
             'uRvsGmrg', 'OxPfh9X4', 'FOzMht1a', 'GludPWCX', 'yULgdCEe', 'rC1EfI6p', 'FEj1cVMl', 'rwGgIZmX', 'OUVNXxMK',
             'A5ktBzKw', 'Wkgf2zYo', 'CAlfe123', '3fNWwuGd', 'oH4Mme1Y', 'Sur0Lxc2', 'wuBrM3fh', 'flzF6Zwx', 'hgaVoSEY',
             'd1pnU4ty', 'uaHYJwZk', 'Dj1UXVy2', 'CBE3mPsk', 'FLMT9dXO', '6JGouKO2', 'OMtLGfjl', 'HW3gFd4U', 'eS2IYRDP',
             'bVuAql95', 'WDJ43dQm', 'h697pPUN', 'aMDr9Jqf', 'JNnTRSEY', 'fjCelvSd', 'Qb210AHL', 'vXNSuWJK', 'hgNkPpSK',
             'uA3yzf5E', '3uyHcYwU', 'pTf5CqcJ', 'AIf2YiqQ', 'AVGgzCE2', 'mGs8RNUK', '5GSCFKkD', 'fpPEcxQ0', 'b5PKlysM',
             'iPmaqeVF', 'mxaykR6h', 'R3IiUxTA', 'yT1OxNg7', 'ABfKYCNt', 'xzlS912e', 'oQkaNMEP', 'qC58NlfW', 'XHz5A4VK',
             '2Kk43hSY', 'TY3cR27W', 'FjtbIRgl', 'Qy7VZ8EW', 'Qbh5iEzv', 'nHKYhNjU', 'KJvfCja5', 'wbRfjKnB', 'bJ2Pvnyk',
             '4O1S8I5J', 'Sq8MxV2T', 'OCsmXLNR', 'YVBsKw1x', 'T8qr4cpN', 'zXBPQEu5', 'MdV0hYWb', 'rMGTHRmh', 'dkeIzRuA',
             'DPsRJAF3', 'yR2BWF5r', 'JEls18i0', 'phzP8MNS', 'vBJ9Nk43', 'ZlxLMN4S', 'RUdel6qP', '1EJShkyv', 'iBxN425a',
             'Tt1exZK7', 'Npks5EAx', 'E4r50xRc', 'yJN2oZXi', 'evD2aSM7', 'rNBItj0M', 'IwvV1kW2', 'XI3TG6gA', 'Wbw8g6Ki',
             'el6nOrYW', 'Fo0ZHjlp', 'G1R6bVoj', 'hPHkAcm1', 'CJoYUe95', 'Uxl1XkKP', 'wxgMNqKa', 'ux1mzj0P', 'ktIJZRvK',
             'wZurGU13', 'ADzM309n', 'M7svIUD2', '3gHFcB7r', 'scTXo8Vp', '98NjCiFz', 'GcCHi0IO', 'DMkqTScu', 'gBdcNH7I',
             'jdfOSea4', '84MgLbpr', 'LC2FypcH', 'kM07cXwK', 'nSlVtTjO', 'HdLbD57V', 'uLtenFHM', 'yBVAXRHd', 'e9hZPwVf',
             'NRUhEYm6', 'vUXTSeOY', 'UycOgH5b', 'wkZvAR5O', 'WqAkZary', 'y9OaRwoH', 'Cvnqsbu4', 'qco9yjFz', 'hyc5kCjM',
             'PXyOub6V', 'sBjVQ8Kp', 'ym60gtiB', '7FVmwt2I', '9QBiT1te', 'GRW4FLot', 'UCFRnusP', '4aX8hwKu', 'tGIabPH7',
             'v14oFuMG', 'KvkTRlBU', 'Xfkw9bHn', 'fSpi43qC', 'iOZL8aKP', 'WJ6nbyXZ', 'zBWJYFfZ', 'o4u1EjcK', 'pbwvtn4T',
             'QH03giK4', '65leZLnc', 'zO0s2Xtg', 'RnugvcBy', 'A6Dw3KNc', 'bKGiZlHm', 'JBcKYNLh', 'yuAqEzTk', 'gNP8ek2x',
             'QHBgUZ2q', 'b98zn0qt', 'mgQh4GtC', 'Wyr5jacX', 'SofWap7J', 'qKkMj6hH', 'dcu6VgYo', 'wXZVYP3q', 'QSCRkxgB',
             '7umcbMfN', '1tP6svJo', 'Vkc1NYOQ', 'c9JDX5WQ', 'dv75TNPi', 'JDntBqCO', 'WxwbU536', '0ZU8VoYa', 'vbyYlLKQ',
             'Oh6D4cTa', 'g56Z7EM1', '0F39itRX', 'Azhn21x9', 'luV8ZoiE', 'dcbmSjUw', 'oKGJqQ9v', 'y4ZDmuWl', 'B4iHXEAo',
             'v6a3FpoX', 'teuhym7O', 's9AlH1tq', 'EfG6Agdy', 'CqigdKmy', 'f7Qb5nCL', 'TQXWdoKy', 'ymGDYZiS', 'lIg4ObSZ',
             'bVeNKXik', 'ePl5usA2', 'KvNmUAqi', 'RPCqtAQU', 'tRK0Ywba', 'JBsp8QeF', '0RncrLqX', 'fcjFWBmi', 'E72aeRIk',
             'gTNXviza', 'tylPOUD3', 'KRy6LghP', 'TQ6IVUr2', 'CErUJsBx', 'hXUlgJGs', 'XLkbZKf8', 'guvhAD3Q', 'zpjgKCt5',
             'uWMzvHs2', 'ZtnC2a4Y', 'qbuoWS4w', 'w3ruHRzK', 'qCdDni06', '1xaM39un', '2iI98tBX', 'bCFpzsfM', 'vKhCDliq',
             'WyVgxvG9', 'v1quzB4o', '67JBi0xZ', 'Hm7MkyfW', 'O9BMtQFn', 'JcHUoAGb', 'RWXSAUf5', 'DHhcY5jm', 'DuZYQPVa',
             'DhV1ytjG', 'UFW6QxoJ', 'fLh6Tp8U', 'Ztrj1eIg', 'Pm4Z8nT0', 'cg9dUrIX', 'ZOEpca41', 'VD7NqfnG', 'rsnxPCTo',
             'JdAWiskP', 'Ru0zDqEx', 'y973UANz', '9bsSpnUY', 'V1hyXikN', 'NcHP1dJF', 'Vmi0LArC', '56ksdejO', 'suEUWTev',
             '3UdW9Zth', 'KOCgEG1S', 'Rmv1uFr8', 's2NPg4eK', 'XMx9SqBT', '5p9ya6Gs', 'j71focDs', 'gc9IOTGh', 'oES6rv50',
             'YkMbvfxl', 'JS4X0H6F', 'XxjzR6bT', 'DQZ9UzYX', 'TqUMFbwG', '2JzMn7AO', 'stydf0Sa', 'RceMBW60', 'n4GO1Ffe',
             'LPVCyHNE', 'tfkNjK3Q', 'XZxVDOp7', 'mXlsdj9o', 'bdoT3ROy', 'okp0LqvC', 'pTnHMBSU', 'XftckyeQ', '4KWwVqvl',
             'ecDkMiBn', '1iK0SUtH', 'zoCZjsG1', 'NBXfUWa4', '83H2uqMG', 'qSChN9wW', 'YHTOhwd8', '1wZ6moMc', 'Mf0CJZUd',
             'OqgaoLRD', 'LlYzyG6J', 'CToWGuFw', 'Ae3cv4QF', 'EATSybCM', '72ramQh3', '9Ynyvh6U', '1YGumCUI', 'Ozdjw3mB',
             'QVruhs6Z', 'oDbermC3', 'zNqOpB2w', 'OTqge0l6', 'MPIwcQjF', 'qhFBSNA8', 'dVvn4Brt', 'SLA6w170', 'PWeFKwYz',
             '9F3i4ywt', 't32yaCuW', '3Dqvkz4R', 'b1M8jkBZ', 'claU5Wzg', 'LHuo6jzT', '456dzDWb', '1YPpDJcT', 'GoI2hmJT',
             'FWMrTQcs', 'bQxYkVKR', 'FyDcAq4I', 'r4amfbzP', 'GgyaekUn', '39QgFHW0', 'hU64gGM9', '0LGfTnQo', 'trGVOMZs',
             'S5qIcZQk', '1TYpxev3', '8BuErVNk', 'buzkZM42', '4Ib6Ocj3', 'Tz0aZ5rP', 'eykVtbuB', '8Ja0Aj5l', 'GOHI9Wa6',
             'ImO69a0n', 'iNAGg49y', 'DgxUr5md', 'ELArxICh', '3EgYp1ke', 'jr5noH9i', 'Ojt9WogS', '0386KGJt', 'EtANaBCV',
             'irWjPAJ3', '7GoxCPgr', 'lfWwBO1u', '35hogzXn', 'IRWUPD39', 'LglJItZa', 'NTgDbs0o', 'nEeGYJ2I', 'nNt4eW6Z',
             '7RntoDz2', 'QsCgej9E', 'iLegsvYy', 'Es9X5K10', 'wPm7HknF', 'lp3FYh2f', 'td3uxJnf', 'rKVH0ZdA', 'Tcj2yzBG',
             'BENgTSsZ', 'HebGyn1z', '0seyQ9mL', '3Da7IsMQ', 'bfSFlLgq', 'gVFiKQGs', '60lhxZO3', 'w47a2HUy', 'iQLFSBa2',
             'wpnF5EB2', 'PUeVDoWK', 'b2wVtRpc', 'RvtXpWQq', 'bNvBXtrc', 'OjYdFrHy', '0gKSbQkj', 'WcNEazGF', '5bRuFtv3',
             'OlRDxBfu', 'UXxZWT0d', 'zDQrXeBU', '9YWJnedM', '6XoZ4TIP', 'vJ2M8hf0', 'vT1X7YGQ', '5hpIT2ZQ', '6mXVbQaC',
             '2ISTAgZe', '0ePCEidF', '08j1ozYb', 'Zm1XuIJb', 'pftEOuaz', '06pWnTAg', 'uxoPdU03', '38wXg5rs', 'py8xTOSr',
             'rUe5hJIX', 'UYN2AVhz', 'g75Mjaiz', 'yYbI1tfe', '60jXieDm', 'wlHxWFCB', 'Mpi3qga7', 'ufKFsQ5J', 'qcM8shPo',
             'Wve3D2Uy', 'GVRjyZk2', 'VDvOYNky', 'Hf4BqFPh', 'YizbOoIr', 'ditWVcTO', 'jPxJ1m5w', 'DG7KV3TN', 'Dx067KFz',
             'KVmRyZF3', 'ayWiOXdv', 'l7WMFHgt', 'RTpeyuft', 'A8JWHeXy', 'GFriHAUu', 'AgvFn3kp', 'YivRI52b', 'LzeIt4J7',
             'qYd5hmIk', 'Z9p7nozH', 'ybA8CTLd', 'IuVcm4sY', 'gyOCs0Do', 'MfqZkhFi', 'dZDoBxRA', 'xa8bqKJe', 'Z6tdkj5f',
             'zfjBHh3Z', '2oQNHCDs', 'cZj93Unl', 'IyeK4X2D', 'nZx9ShT4', '3x4z8fij', '84kbVuty', 'EVWwJc43', 'L5P1mtNC',
             'R64PGkrW', 'ZkRhj4Fu', 'QwtTBxDR', 'LstD9ZBr', 'ufA0BH4n', 'FxE3yUpO', 'YNr08lKH', 'IlXtmg53', 'P4IBku6H',
             'UOGHtrQf', 'HkovyB04', 'gXi1GYAU', 'fbYQI13l', 'J5Ur1PWC', 'Ds4Z7JuE', 'P4e0mkIA', '60N7ZPOT', 'mvlPOwA4',
             'TQkuEJ2s', 'm9WHKGZ5', '70qkTNdP', 'YlM7TWfU', 'UTQF3Wva', 'nidOEVTJ', '90XjFDxK', 'qvy4e37P', 'PSg8jQwD',
             'sRopOhuF', 'OdKtPZiI', 'RwV7QOmI', 'YuDa25Cr', 'w3jiE6V2', 'jT2EKAm3', 'Unzc9soX', 'vEt6NkBT', 'Nexklyum',
             'w9A5IMXt', 'O4zA7mCu', 'NFpygWAC', 'iBYAgU92', '5Jmk67Ch', '19GYH2hg', 'BjOxaKzF', 'culXwf6P', 'Otr71DAe',
             'oMEfuVsk', 'l1mRHBoE', 'JciYM5rs', 'JZ3zVPQ9', 'wjq61zUn', 'vlw8Rd3K', 'vG4Nup06', 'X2NIEFAg', 'zescQaXK',
             'QuNYvS3x', 'JAckiBS6', 'A81CjHV9', '3umTvQlo', 'hsE5yv26', 'bEhtcCLv', '7lOBHG2w', '5EPLMDoc', 'OEFwj6DH']

    labels = [1, 3, 2, 1, 3, 1, 1, 0, 1, 3, 1, 2, 2, 1, 1, 3, 0, 2, 2, 1, 1, 1, 0, 3, 3, 3, 2, 3, 3, 2, 0, 1, 0, 1, 1, 0, 3,
              3, 3, 1, 1, 3, 1, 3, 1, 2, 3, 1, 0, 2, 2, 1, 3, 3, 2, 2, 1, 1, 2, 1, 3, 0, 2, 1, 2, 1, 2, 3, 3, 3, 2, 1, 0, 2,
              1, 2, 1, 3, 0, 1, 3, 1, 1, 1, 1, 2, 3, 3, 2, 3, 0, 3, 1, 0, 3, 2, 3, 1, 2, 1, 3, 2, 3, 2, 1, 1, 1, 1, 1, 3, 3,
              1, 1, 2, 1, 3, 3, 3, 1, 1, 2, 1, 3, 2, 1, 2, 1, 2, 2, 0, 3, 2, 1, 1, 2, 2, 1, 2, 1, 3, 2, 3, 3, 1, 3, 2, 1, 3,
              1, 1, 1, 0, 2, 1, 2, 1, 3, 2, 3, 3, 0, 2, 2, 1, 1, 1, 2, 2, 1, 2, 3, 1, 0, 2, 3, 2, 3, 1, 1, 3, 3, 1, 1, 0, 2,
              1, 1, 2, 2, 0, 1, 3, 0, 1, 0, 3, 3, 1, 3, 0, 2, 1, 0, 2, 3, 1, 2, 1, 1, 0, 0, 1, 2, 3, 2, 1, 1, 1, 3, 1, 3, 1,
              2, 1, 1, 2, 1, 1, 3, 2, 2, 2, 1, 2, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 2, 3, 2, 1, 2, 1, 3, 1, 1, 3, 2, 2, 3, 1, 0,
              2, 1, 2, 0, 3, 0, 2, 2, 3, 3, 1, 1, 3, 2, 3, 1, 3, 3, 1, 1, 2, 0, 3, 3, 2, 2, 1, 2, 3, 0, 2, 1, 3, 2, 1, 3, 1,
              1, 2, 1, 0, 2, 1, 3, 0, 0, 3, 1, 0, 0, 3, 2, 3, 3, 2, 3, 2, 3, 1, 2, 3, 3, 1, 3, 2, 0, 1, 3, 2, 0, 3, 2, 3, 3,
              1, 0, 2, 2, 2, 2, 3, 3, 2, 3, 1, 2, 2, 0, 0, 0, 2, 2, 2, 1, 1, 2, 1, 2, 1, 3, 2, 2, 1, 2, 1, 0, 1, 1, 3, 3, 3,
              0, 2, 3, 1, 1, 1, 2, 3, 3, 3, 2, 2, 2, 1, 2, 1, 1, 3, 3, 0, 1, 1, 3, 1, 3, 1, 2, 3, 0, 1, 3, 0, 0, 2, 2, 3, 2,
              2, 2, 3, 1, 2, 1, 1, 0, 2, 1, 3, 3, 1, 2, 2, 2, 2, 0, 3, 3, 0, 3, 2, 2, 3, 1, 3, 0, 1, 1, 1, 1, 3, 1, 3, 1, 1,
              2, 3, 2, 2, 0, 0, 2, 2, 1, 3, 2, 1, 3, 2, 3, 2, 3, 1, 3, 2, 1, 1, 2, 2, 2, 3, 0, 3, 3, 1, 1, 1, 2, 3, 3, 2, 3,
              1, 3, 3, 3, 1, 0, 3, 2, 1, 1, 0, 0, 3, 1, 1, 3, 1, 2, 3, 1, 2, 0, 1, 1, 1, 1, 1, 2, 0, 1, 2, 1, 3, 1, 0, 2, 1,
              1, 2, 3, 1, 2, 1, 3, 2, 2, 3, 0, 2, 2, 2, 2, 3, 1, 2, 3, 1, 3, 2, 2, 0, 0, 1, 1, 2, 3, 2, 2, 3, 1, 2, 1, 3, 3,
              3, 1, 1, 2, 2, 1, 2, 1, 1, 3, 2, 1, 0, 1, 0, 3, 2, 3, 1, 1, 0, 3, 2, 3, 1, 1, 1, 1, 0, 0, 3, 1, 1, 2, 2, 1, 2,
              1, 1, 1, 1, 0, 3, 0, 1, 0, 3, 0, 1, 3, 1, 1, 0, 1, 3, 0, 2, 1, 2, 1, 0, 1, 2, 2, 3, 1, 1, 1, 1, 1, 3, 2, 0, 1,
              2, 3, 3, 1, 1, 0, 1, 1, 2, 0, 0, 1, 2, 1, 1, 2, 0, 3, 2, 2, 1, 3, 1, 3, 2, 1, 3, 1, 2, 1, 2, 2, 2, 3, 3, 3, 3,
              2, 3, 3, 1, 1, 2, 2, 1, 2, 2, 2, 1, 1, 3, 1, 1, 1, 0, 3, 1, 1, 1, 2, 2, 3, 1, 2, 3, 2, 2, 0, 3, 1, 1, 2, 1, 3,
              3, 0, 3, 2, 2, 2, 1, 1, 3, 2, 3, 0, 0, 0, 1, 2, 3, 1, 1, 3, 2, 2, 2, 0, 2, 1, 0, 1, 2, 3, 1, 2, 1, 0, 2, 1, 2,
              2, 2, 3, 1, 3, 1, 2, 3, 1, 3, 2, 3, 0, 3, 2, 1, 1, 2, 1, 1, 1, 0, 0, 3, 2, 2, 1, 1, 1, 3, 1, 1, 1, 1, 0, 3, 3,
              3, 1, 2, 1, 3, 3, 1, 3, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 1, 1, 1, 1, 1, 1, 2, 0, 0, 0, 2, 1, 3, 0, 0, 1, 1, 2, 3,
              2, 3, 0, 0, 1, 2, 1, 3, 0, 3, 1, 3, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 2, 1, 1, 2, 1, 3, 2, 1, 1, 2, 3, 3, 2, 1, 1,
              3, 2, 3, 1, 1, 3, 0, 1, 3, 2, 0, 3, 2, 3, 3, 1, 2, 3, 1, 2, 1, 1, 3, 2, 2, 1, 1, 2, 3, 1, 0, 2, 1, 1, 3, 1, 3,
              0, 1, 1, 1, 0, 1, 2, 0, 1, 1, 3, 0, 1, 1, 1, 2, 0, 3, 1, 1, 1, 1, 2, 1, 0, 2, 0, 1, 2, 1, 2, 2, 1, 3, 2, 2, 2,
              2, 1, 1, 1, 1, 1, 2, 0, 1, 2, 3, 3, 0, 0, 3, 1, 2, 1, 3, 0, 1, 3, 2, 1, 0, 3, 1, 1, 3, 2, 0, 3, 2, 1, 3, 1, 0,
              3, 2, 0, 1, 3, 1, 3, 1, 3, 1, 2, 3, 3, 0, 1, 3, 3, 2, 1, 3, 2, 1, 2, 1, 0, 1, 1, 2, 2, 2, 1, 2, 2, 1, 0, 0, 1,
              1, 2, 3, 3, 1, 3, 1, 1, 1, 2, 2, 1, 2, 3, 2, 1, 2, 2, 1, 2, 2, 1, 1, 3, 3, 1, 1, 0, 2, 2, 2, 3, 1, 0, 0, 1, 3,
              1, 3, 3, 2, 2, 1, 0, 0, 2, 3, 1, 0, 2, 2, 2, 2, 1, 1, 3, 2, 1, 3, 3, 1, 3, 3, 2, 2, 3, 3, 2, 3, 0, 3, 2, 3, 2,
              2, 1, 3, 3, 3, 0, 3, 1, 1, 3, 0, 1, 1, 2, 2, 3, 2, 1, 2, 3, 3, 2, 1, 1, 3, 1, 0, 0, 2, 2, 1, 1, 1, 1, 1, 0, 1,
              2, 2, 1, 3, 2, 2, 2, 3, 1, 1, 1, 2, 3, 3, 1, 2, 0, 2, 1, 1, 3, 0, 1, 1, 3, 1, 1, 1, 2, 1, 3, 2, 1, 2, 3, 3, 1,
              1, 1, 3, 3, 2, 1, 2, 2, 2, 2, 2, 2, 0, 3, 2, 2, 1, 3, 0, 2, 1, 1, 0, 2, 1, 3, 1, 2, 1, 3, 3, 1, 3, 2, 0, 3, 1,
              3, 1, 1, 2, 0, 1, 2, 2, 2, 2, 2, 1, 3, 1, 3, 2, 1, 2, 3, 2, 1, 3, 2, 0, 0, 3, 1, 2, 2, 2, 2, 0, 2, 1, 2, 1, 2,
              1, 0, 1, 2, 1, 1, 3, 3, 3, 2, 2, 1, 3, 2, 2, 1, 1, 1, 1, 3, 1, 2, 1, 0, 3, 1, 2, 1, 1, 2, 1, 3, 0, 1, 1, 1, 2,
              3, 2, 3, 1, 1, 3, 2, 1, 1, 1, 3, 3, 3, 2, 2, 2, 3, 0, 2, 1, 3, 2, 2, 2, 1, 2, 2, 3, 1, 1, 2, 1, 2, 1, 1, 0, 1,
              2, 3, 1, 1, 1, 3, 3, 1, 1, 2, 1, 0, 3, 1, 0, 1, 2, 3, 1, 1, 3, 2, 1, 1, 0, 3, 1, 1, 2, 1, 0, 3, 0, 2, 1, 3, 0,
              2, 2, 1, 1, 1, 2, 1, 0, 2, 2, 1, 1, 0, 2, 3, 0, 3, 2, 1, 2, 1, 1, 1, 2, 3, 1, 1, 1, 1, 0, 2, 2, 3, 1, 1, 3, 2,
              1, 3, 0, 3, 3, 1, 1, 2, 1, 3, 2, 2, 3, 1, 3, 3, 2, 3, 3, 1, 1, 0, 0, 2, 2, 3, 2, 0, 3, 2, 3, 2, 0, 1, 2, 1, 1,
              0, 2, 1, 0, 1, 3, 1, 1, 3, 1, 1, 0, 1, 3, 3, 1, 1, 1, 2, 0, 1, 1, 1, 3, 1, 3, 1, 0, 2, 1, 2, 1, 1, 2, 1, 1, 1,
              3, 1, 2, 3, 1, 1, 3, 2, 1, 1, 2, 1, 3, 1, 1, 1, 1, 1, 2, 1, 0, 1, 2, 3, 3, 0, 1, 0, 1, 0, 1, 3, 1, 0, 2, 2, 2,
              2, 0, 3, 1, 3, 3, 2, 2, 3, 2, 1, 1, 1, 1, 1, 1, 2, 0, 2, 2, 3, 1, 1, 1, 2, 0, 3, 3, 1, 3, 0, 3, 0, 1, 2, 1, 1,
              3, 0, 2, 0, 3, 2, 0, 2, 2, 1, 2, 0, 1, 0, 2, 2, 2, 0, 3, 1, 1, 1, 2, 2, 1, 1, 1, 1, 3, 1, 2, 0, 3, 3, 0, 2, 2,
              3, 3, 1, 3, 1, 3, 1, 3, 2, 3, 1, 1, 0, 3, 2, 3, 2, 0, 1, 1, 1, 0, 0, 0, 3, 3, 3, 0, 0, 1, 0, 3, 1, 3, 1, 1, 1,
              2, 1, 0, 0, 2, 3, 2, 3, 1, 1, 3, 3, 3, 0, 2, 0, 2, 3, 1, 3, 1, 2, 2, 1, 0, 3, 0, 1, 1, 1, 3, 1, 3, 2, 2, 1, 0,
              2, 0, 2, 2, 1, 2, 2, 1, 1, 2, 3, 2, 0, 1, 2, 3, 0, 3, 0, 3, 2, 1, 1, 2, 2, 2, 0, 3, 3, 3, 1, 2, 2, 0, 3, 1, 2,
              1, 1, 2, 3, 2, 1, 1, 1, 3, 1, 0, 1, 0, 2, 0, 1, 2, 0, 2, 1, 1, 3, 0, 1, 3, 1, 2, 0, 1, 1, 1, 3, 1, 0, 2, 2, 2,
              1, 0, 1, 3, 3, 2, 2, 1, 1, 3, 3, 0, 1, 2, 0, 1, 0, 1, 2, 1, 0, 3, 2, 1, 3, 1, 3, 3, 1, 3, 1, 2, 2, 1, 1, 1, 1,
              3, 1, 3, 3, 1, 2, 3, 1, 2, 1, 3, 0, 2, 2, 2, 0, 0, 2, 0, 3, 3, 3, 1, 1, 1, 1, 3, 2, 3, 2, 3, 1, 1, 3, 2, 3, 3,
              1, 0, 2, 3, 1, 2, 3, 3, 2, 3, 1, 3, 3, 3, 2, 3, 1, 2, 1, 1, 1, 2, 0, 1, 3, 3, 3, 2, 0, 1, 1, 1, 2, 2, 2, 1, 0,
              3, 1, 2, 1, 3, 1, 1, 1, 1, 2, 1, 3, 3, 1, 3, 2, 2, 1, 0, 1, 2, 1, 2, 0, 3, 1, 2, 1, 3, 1, 3, 2, 1, 1, 2, 3, 0,
              1, 1, 2, 2, 2, 0, 1, 3, 1, 3, 0, 3, 1, 1, 1, 0, 2, 1, 2, 3, 1, 3, 2, 2, 0, 1, 3, 0, 0, 1, 3, 0, 3, 1, 2, 2, 3,
              2, 0, 2, 2, 3, 3, 1, 2, 1, 3, 2, 1, 2, 1, 0, 3, 2, 2, 1, 3, 0, 3, 2, 2, 3, 1, 3, 3, 2, 3, 1, 2, 1, 1, 2, 1, 1,
              3, 2, 2, 1, 3, 0, 3, 1, 2, 2, 3, 2, 1, 3, 2, 1, 1, 3, 3, 0, 2, 2, 0, 2, 2, 1, 2, 0, 3, 2, 2, 1, 2, 1, 2, 3, 1,
              3, 2, 3, 2, 2, 3, 3, 1, 3, 3, 1, 1, 3, 2, 1, 1, 3, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 3, 0, 3, 1, 2, 2,
              1, 2, 1, 1, 1, 1, 1, 0, 2, 1, 1, 2, 2, 2, 2, 1, 3, 0, 1, 1, 2, 0, 1, 0, 3, 1, 2, 0, 0, 3, 1, 1, 1, 1, 1, 1, 2,
              2, 1, 2, 1, 3, 3, 0, 2, 1, 3, 1, 3, 0, 3, 2, 3, 1, 3, 0, 1, 2, 3, 2, 1, 1, 0, 0, 1, 1, 1, 2, 1, 3, 2, 3, 1, 2,
              2, 2, 3, 3, 1, 0, 1, 3, 2, 2, 1, 2, 2, 1, 2, 1, 2, 0, 1, 1, 3, 1, 3, 1, 1, 1, 2, 1, 2, 2, 2, 0, 3, 2, 0, 2, 2,
              1, 3, 3, 2, 2, 2, 2, 3, 3, 2, 1, 1, 2, 0, 2, 1, 1, 0, 0, 1, 1, 0, 1, 2, 0, 2, 0, 1, 3, 1, 3, 2, 2, 0, 2, 3, 2,
              1, 1, 2, 0, 0, 0, 2, 2, 1, 3, 2, 3, 2, 2, 2, 2, 0, 3, 2, 0, 3, 1, 2, 3, 2, 1, 3, 2, 3, 3, 2, 1, 3, 3, 1, 1, 1,
              2, 2, 1, 2, 1, 1, 3, 3, 0, 1, 3, 1, 3, 2, 0, 3, 1, 2, 0, 1, 3, 1, 1, 1, 2, 3, 2, 2, 0, 2, 1, 2, 1, 3, 2, 3, 2,
              2, 3, 3, 1, 0, 2, 2, 3, 1, 1, 0, 2, 3, 2, 2, 1, 1, 2, 2, 2, 0, 1, 1, 1, 3, 1, 3, 3, 1, 2, 1, 3, 1, 2, 3, 2, 2,
              1, 1, 1, 3, 1, 2, 1, 3, 3, 2, 1, 2, 1, 1, 0, 1, 1, 1, 2, 2, 3, 3, 0, 1, 2, 2, 1, 3, 2, 1, 0, 3, 2, 1, 2, 2, 1,
              0, 2, 0, 2, 2, 2, 2, 3, 3, 1, 2, 0, 2, 2, 3, 1, 0, 0, 1, 2, 3, 1, 0, 2, 1, 1, 3, 3, 3, 2, 1, 2, 3, 3, 3, 3, 3,
              3, 3, 3, 1, 2, 0, 0, 3, 3, 1, 2, 3, 1, 2, 2, 0, 1, 3, 0, 1, 1, 2, 2, 2, 2, 1, 2, 3, 2, 1, 2, 2, 1, 1, 3, 3, 3,
              2, 0, 0, 1, 0, 3, 2, 2, 1, 1, 3, 3, 3, 3, 0, 1, 1, 2, 2, 1, 1, 2, 3, 1, 2, 0, 1, 2, 2, 3, 0, 2, 3, 2, 1, 1, 2,
              1, 2, 1, 1, 0, 1, 1, 2, 2, 2, 3, 2, 3, 1, 1, 2, 3, 1, 3, 3, 1, 3, 2, 1, 2, 3, 0, 1, 1, 1, 1, 2, 1, 3, 3, 3, 1,
              1, 3, 1, 2, 3, 2, 1, 2, 1, 2, 3, 1, 2, 1, 3, 0, 0, 0, 2, 1, 3, 0, 1, 3, 2, 2, 1, 2, 3, 3, 3, 2, 3, 2, 1, 2, 2,
              0, 3, 3, 3, 1, 3, 1, 1, 3, 3, 3, 2, 2, 3, 2, 0, 2, 1, 2, 2, 0, 2, 1, 2, 1, 3, 2, 3, 3, 1, 1, 2, 1, 3, 3, 2, 1,
              1, 1, 1, 1]

    print('Start loading data.')

    # all_imgs = [''.join([os.path.join(train_imgs, str(i) + '.png')]) for i in names][1:100]
    masks = [''.join([os.path.join(train_masks, str(i) + '.png')]) for i in names]

    # imgs = [cv2.resize(cv2.imread(i), (res, res))[:, :, ::-1] for i in all_imgs]
    masks = [cv2.imread(i)[:, :, -1] for i in masks]
    import multiprocessing as mp

    multi_count(masks)
    a =1
    return



    aug_size = int(res / 10)
    train_transform = A.Compose([
        A.RandomRotate90(),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.2, 0.1), rotate_limit=40, p=0.5),
        A.RandomBrightnessContrast(
            brightness_limit=0.5,
            contrast_limit=0.1,
            p=0.5
        ),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=100, val_shift_limit=80),
        A.OneOf([
            A.CoarseDropout(max_holes=100, max_height=aug_size, max_width=aug_size, fill_value=[239, 234, 238]),
            A.GaussNoise()
        ]),
        A.OneOf([
            A.ElasticTransform(),
            A.GridDistortion(),
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0)
        ]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()], p=1.)
    test_transform = A.Compose([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            max_pixel_value=255.0,
            p=1.0
        ),
        ToTensorV2()], p=1.)





    def fit(epoch, model, trainloader, testloader,cv_corr, cv_tot):
        # cv_corr, cv_tot = 0,0

        with tqdm(total=len(trainloader), ncols=120, ascii=True) as t:

            running_loss = 0
            model.train()
            correct = 0
            total = 0
            for batch_idx, (imgs, masks) in enumerate(trainloader):
                t.set_description("Train(Epoch{}/{})".format(epoch + 1, epochs))
                imgs, masks_cuda = imgs.to('cuda'), masks.to('cuda')
                imgs = imgs.float()
                with autocast():
                    masks_pred = model(imgs)
                    loss = criterion(masks_pred, masks_cuda)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                with torch.no_grad():
                    predicted = masks_pred.argmax(1)
                    corr, tot = Pa(predicted, masks_cuda)
                    correct += corr
                    total += tot
                    epoch_acc = correct / total
                    running_loss += loss.item()
                t.set_postfix(loss='{:.3f}'.format(running_loss / (batch_idx + 1)),
                              train_pa='{:.2f}%'.format(epoch_acc * 100))
                t.update(1)
            # epoch_acc = correct / total
            epoch_loss = running_loss / len(trainloader.dataset)
        with tqdm(total=len(testloader), ncols=120, ascii=True) as t:
            test_running_loss = 0
            val_correct = 0
            val_total = 0
            model.eval()
            with torch.no_grad():
                for batch_idx, (imgs, masks) in enumerate(testloader):
                    t.set_description("val(Epoch{}/{})".format(epoch + 1, epochs))
                    imgs, masks_cuda = imgs.to('cuda'), masks.to('cuda')
                    imgs = imgs.float()
                    masks_pred = TTA(model, imgs)
                    predicted = masks_pred.argmax(1)
                    # print(masks_cuda.shape)
                    predicted = F.interpolate(torch.unsqueeze(predicted, 1).float(), size=masks_cuda.shape[1:],
                                              mode='nearest')
                    corr, tot = Pa(predicted, masks_cuda)

                    val_correct += corr
                    cv_corr += corr
                    val_total += tot
                    cv_tot += tot
                    epoch_test_acc = val_correct / val_total

                    t.set_postfix(val_pa='{:.2f}%'.format(epoch_test_acc * 100))

                    t.update(1)

            CosineLR.step(epoch)

            return epoch_loss, epoch_acc.item(), epoch_test_acc.item(), val_correct, val_total, cv_corr, cv_tot


    c = int(len(imgs) * 0.2)
    after_read_date = time.time()
    print('data_time', after_read_date - begin_time)


    ns_encoder_path = '/home/project/temp_data/tf_efficientnet_b3_ns-9d44bf68.pth'
    imagenet_encoder_path = '/home/project/temp_data/tf_efficientnet_b3_aa-84b4657e.pth'


    for i in range(5):
        cv_corr = 0
        cv_tot = 0
        best_acc_final = []

        cv_best_corr = 0
        cv_best_total = 0

        print('Fold {:} start training'.format(i))
        print("training %s"%model_name)
        if model_name == 'unpp_ns':
            model = smp.UnetPlusPlus(encoder_name='timm-efficientnet-b3', encoder_weights=None, classes=2)
            model.encoder.load_state_dict(torch.load(ns_encoder_path))
        elif model_name == 'FPN_ns':
            model = smp.FPN(encoder_name='timm-efficientnet-b3', encoder_weights=None, classes=2)
            model.encoder.load_state_dict(torch.load(ns_encoder_path))
        elif model_name == 'un_scse':
            model = smp.Unet(encoder_name='timm-efficientnet-b3', encoder_weights=None, decoder_attention_type='scse',
                             classes=2)
            model.encoder.load_state_dict(torch.load(imagenet_encoder_path))
        for para in model.encoder.parameters():
            para.requires_grad = False

        train_imgs = imgs[:i * c] + imgs[(i + 1) * c:]
        train_masks = masks[:i * c] + masks[(i + 1) * c:]
        test_imgs = imgs[i * c:(i + 1) * c]
        test_masks = masks[i * c:(i + 1) * c]

        train_ds = train_dataset(train_imgs, train_masks, train_transform)
        test_ds = test_dataset(test_imgs, test_masks, test_transform)

        criterion = nn.CrossEntropyLoss(reduction='sum')
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        CosineLR = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=warmup_epochs,
                                                                        T_mult=(epochs // warmup_epochs))

        train_dl = DataLoader(
            train_ds,
            shuffle=True,
            batch_size=BATCH_SIZE,
            pin_memory=False,
            num_workers=4,
            drop_last=True,
        )

        test_dl = DataLoader(
            test_ds,
            batch_size=1,
            pin_memory=False,
            num_workers=4,
        )

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0

        train_loss = []
        train_acc = []
        test_loss = []
        test_acc = []

        scaler = GradScaler()
        if torch.cuda.is_available():
            model.to('cuda')

        for epoch in range(epochs):
            if epoch == warmup_epochs:
                for para in model.encoder.parameters():
                    para.requires_grad = True

            epoch_loss, epoch_acc, epoch_test_acc, val_correct, val_total,cv_corr,cv_tot = fit(epoch, model, train_dl, test_dl,cv_corr, cv_tot)

            if epoch_test_acc > best_acc:
                best_model_wts = copy.deepcopy(model.state_dict())
                best_acc = epoch_test_acc
                fold_best_corr = val_correct
                fold_best_total = val_total
                torch.save(best_model_wts, os.path.join('/home/project/model', save_name + '_fold_' + str(i) + '.pth', ))

            train_loss.append(epoch_loss)
            train_acc.append(epoch_acc)
            test_acc.append(epoch_test_acc)
            if (epoch_test_acc <= epoch_acc) and (epoch_test_acc > 80.0):
                break

        torch.cuda.empty_cache()

        cv_corr += fold_best_corr
        cv_tot += fold_best_total

        print('Fold {:} trained successfully. Best AP:{:5f}'.format(i, best_acc))


        # plt.savefig(''.join([save_name, '_fold_', str(i), '_Acc.png']), bbox_inches='tight')

    after_net_time = time.time()
    print('Net:{:} Time:{:5f} CV_PA:{:5f}'.format(save_name, (after_net_time - after_read_date), (cv_corr.item() / cv_tot)))
if __name__ == '__main__':
    train('unpp_ns')
