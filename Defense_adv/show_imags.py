from AAADefense_mode import VAE
import torch.utils.data as Data
import torch
import os
import numpy as np
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from torchvision.utils import save_image
from torchvision.utils import make_grid
from advertorch.attacks.jsma import JacobianSaliencyMapAttack
import torchvision
import torchvision.transforms as transforms
from attack_model import victim

test_data = torchvision.datasets.MNIST(
    root='../data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
)
test_loader = Data.DataLoader(test_data, batch_size=1, num_workers=0, shuffle=True)

device = 'cuda'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

defense_mode = VAE()
defense_mode = defense_mode.to(device)
defense_mode.load_state_dict(torch.load("../mode_save/CVAE-GAN-VAE.pth"))
defense_mode.eval()

victim_mode = victim().to(device)
victim_mode.load_state_dict(torch.load("../LeNet_5.pth"))
victim_mode.eval()
adversary = JacobianSaliencyMapAttack(victim_mode, num_classes=10)

count = 0
total = 0


for x, y in test_loader:
    x, y = x.to(device), y.to(device)
    t = torch.tensor([6]).to(device)
    t1 = torch.tensor([4]).to(device)

    t = t.to(device)
    if y == t1:
        x_jsma = adversary.perturb(x, t)
        x_pgd = projected_gradient_descent(victim_mode, x, 0.7, 0.01, 40, np.inf)
        jsma_rec = defense_mode.rec_smi(x_jsma)
        pgd_rec = defense_mode.rec_smi(x_pgd)

        real_images = make_grid(x.cpu(), nrow=8, normalize=True).detach()
        save_image(real_images, '.real_images-{}.png'.format(int(y)))

        real_images = make_grid(x_jsma.cpu(), nrow=8, normalize=True).detach()
        save_image(real_images, '.jsma_images-{}.png'.format(int(y)))

        real_images = make_grid(x_pgd.cpu(), nrow=8, normalize=True).detach()
        save_image(real_images, '.pgd_images-{}.png'.format(int(y)))

        real_images = make_grid(pgd_rec.cpu(), nrow=8, normalize=True).detach()
        save_image(real_images, '.pgd_rec_images-{}.png'.format(int(y)))

        real_images = make_grid(jsma_rec.cpu(), nrow=8, normalize=True).detach()
        save_image(real_images, '.jsma_rec_images-{}.png'.format(int(y)))

        break


