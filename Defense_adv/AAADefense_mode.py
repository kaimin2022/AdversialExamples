from torch import nn
import torch
import os
import torch.utils.data as Data
import torchvision
from torchvision.transforms import transforms

device = 'cuda'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
nz = 200
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # 定义编码器
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.encoder_fc1 = nn.Linear(32 * 7 * 7, nz)
        self.encoder_fc2 = nn.Linear(32 * 7 * 7, nz)
        self.Sigmoid = nn.Sigmoid()
        self.decoder_fc = nn.Linear(nz, 32 * 7 * 7)
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def noise_reparameterize(self, mean, logvar):
        eps = torch.randn(mean.shape).to(device)
        z = mean + eps * torch.exp(logvar)
        return z

    def forward(self, x):
        z = self.encoder(x)
        output = self.decoder(z)
        return output

    def encoder(self, x):
        out1, out2 = self.encoder_conv(x), self.encoder_conv(x)
        # out1:(b, 32, 7, 7)  z:(b, 200)
        mean = self.encoder_fc1(out1.view(out1.shape[0], -1))
        logstd = self.encoder_fc2(out2.view(out2.shape[0], -1))
        z = self.noise_reparameterize(mean, logstd)
        # print("---", out1.shape, z.shape)
        return z, mean, logstd

    def decoder(self, z):
        out3 = self.decoder_fc(z)
        out3 = out3.view(out3.shape[0], 32, 7, 7)
        # out3:(b, 32, 7, 7)
        out3 = self.decoder_deconv(out3)
        return out3

    def rec_smi(self, x):
        x = self.encoder_conv(x)
        out = self.decoder_deconv(x)
        return out
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(7*7*64, 200)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(200, 84)
        self.relu4 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(84, 10)

    def forward(self, x):
        out = self.maxpool1(self.relu1(self.conv1(x)))
        out = self.maxpool2(self.relu2(self.conv2(out)))
        out = out.view(out.size(0), -1)
        out = self.relu3(self.linear1(out))
        out = self.relu4(self.linear2(out))
        out = self.linear3(out)
        return out

defense_mode = VAE()
defense_mode = defense_mode.to(device)
defense_mode.load_state_dict(torch.load("../mode_save/CVAE-GAN-VAE.pth"))
defense_mode.eval()

mode = LeNet5()
mode = mode.to(device)
mode.load_state_dict(torch.load("../LeNet_5.pth"))
mode.eval()

test_data = torchvision.datasets.MNIST(
    root='../data', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
)
test_data_loader = Data.DataLoader(test_data, batch_size=128, num_workers=0, shuffle=False)
count = 0
total = 0
# for x, y in test_data_loader:
#     x = x.to(device)
#     y = y.to(device)
#     output = defense_mode.rec_smi(x)
#     _, pre = mode(output).max(1)
#     count += pre.eq(y).sum().item()
#     total += y.size(0)
# print(count, total, 1-count/total)