import json
import os
import numpy as np
import re
import torch
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from torchvision.utils import save_image
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transform
device = 'cuda'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def _load_data(dir):
    # loay x
    files = os.listdir(dir)

    idFiles = [(int(re.match('x_(.*?).npy', fn).group(1)), fn) for fn in files]
    idFiles.sort(key=lambda x: x[0])

    arrayFns = [os.path.join(dir, fn) for _, fn in idFiles]
    arrays = [np.load(fn) for fn in arrayFns]
    x = np.concatenate(arrays, axis=0)

    # load y
    y = np.load('y.npy')
    return x, y
def load_robust_dataset():
    return _load_data('robust')
robust = load_robust_dataset()
# print(robust[0].shape)  # data (60000, 28, 28)
# print(robust[1].shape)  # label (60000)
data = robust[0]  # Numpy
label = robust[1]
data = torch.from_numpy(data)
label = torch.from_numpy(label)
train_data = Data.TensorDataset(data, label)

# 切分Robust数据集
R_train, R_test = train_test_split(
    train_data, test_size=1 / 6, random_state=123
)
transforms = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(mean=(0.1307,), std=(0.3081,))
])
Robust_transforms = transform.Compose([
    transform.Normalize(mean=(0.1307,), std=(0.3081,))
])
# Natural数据集
Nature_train = torchvision.datasets.MNIST(root='../data', train=True, download=True, transform=transforms)
Nature_test = torchvision.datasets.MNIST(root='../data', train=False, download=True, transform=transforms)

Robust_test_loader = DataLoader(  # 10000
    dataset=R_test,
    batch_size=1,
    shuffle=True
)
count = 0
for x, y in Nature_test:
    x = x.to(device)
    if count < 5:

        if y == torch.tensor([3]):
            count += 1
            real_images = make_grid(x.cpu(), nrow=8, normalize=True).detach()
            save_image(real_images, './images-{}.png'.format(int(y)))
        
        if y == torch.tensor([9]):
            count += 1
            real_images = make_grid(x.cpu(), nrow=8, normalize=True).detach()
            save_image(real_images, './images-{}.png'.format(int(y)))
    else:
        break




# for index,(x, y) in enumerate(Robust_test_loader):
#     x = x.to(device)
#     x = x.unsqueeze(dim=1)
#     y = y.to(device)
#     if index < 5:
#         real_images = make_grid(x.cpu(), nrow=8, normalize=True).detach()
#         save_image(real_images, './robust_images-{}.png'.format(int(y)))
#     else:
#         break