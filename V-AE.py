from data_for_Vae.get_triple_dataloader import train_loader, test_loader
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision.utils import save_image
from torchvision.utils import make_grid
import os


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


class Discriminator(nn.Module):
    def __init__(self, outputn=1):
        super(Discriminator, self).__init__()
        self.dis = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 2)),
        )
        self.fc = nn.Sequential(
            nn.Linear(7 * 7 * 64, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, outputn),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.dis(input)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(1)


def loss_function(recon_x, x, mean, logstd):
    # BCE = F.binary_cross_entropy(recon_x,x,reduction='sum')
    MSE = MSECriterion(recon_x, x)
    # 因为var是标准差的自然对数，先求自然对数然后平方转换成方差
    var = torch.pow(torch.exp(logstd), 2)
    KLD = -0.5 * torch.sum(1 + torch.log(var) - torch.pow(mean, 2) - var)
    return 0.8 * MSE + 0.2 * KLD


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(2)
        self.linear1 = nn.Linear(7 * 7 * 64, 200)
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


class LR(nn.Module):
    def __init__(self):
        super(LR, self).__init__()
        self.layer1 = nn.Linear(200, 2)

    def forward(self, x):
        x = self.layer1(x)
        return x


batchSize = 256
imageSize = 28
nz = 200
nepoch = 2000

mode = LeNet5()
mode.load_state_dict(torch.load("LeNet_5.pth"))
mode.eval()

D_lat = LR()
D_lat.load_state_dict(torch.load("data_for_discrim/Train_D/Dis_AE(x).pth"))
D_lat.eval()
if not os.path.exists('./img_CVAE-GAN'):
    os.mkdir('./img_CVAE-GAN')
device = 'cuda'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
mode.to(device)
D_lat.to(device)
cudnn.benchmark = True
best_acc = 0
print("=====> 构建VAE")
vae = VAE().to(device)
print("=====> 构建D")
D = Discriminator(1).to(device)
criterion = nn.BCELoss().to(device)
cri = nn.CrossEntropyLoss()
MSECriterion = nn.MSELoss().to(device)
print("=====> Setup optimizer")
optimizerD = optim.Adam(D.parameters(), lr=0.0001)
optimizerVAE = optim.Adam(vae.parameters(), lr=0.0001)


def train(epoch):
    count = 0
    total = 0
    vae.train()
    D.train()
    for i, (pgd, data, label) in enumerate(train_loader):
        # print(pgd.shape, data.shape)
        # 先处理数据
        pgd = pgd.to(device)
        data = data.to(device)
        label = label.to(device)
        batch_size = data.shape[0]
        # 训练D
        output = D(data)
        real_label = torch.ones(batch_size).to(device)  # 定义真实的图片label为1
        fake_label = torch.zeros(batch_size).to(device)  # 定义假的图片的label为0
        errD_real = criterion(output, real_label)

        fake_data = vae.rec_smi(pgd)
        output = D(fake_data)
        errD_fake = criterion(output, fake_label)
        # new add
        output1 = D(pgd)
        errD_fake1 = criterion(output1, fake_label)

        errD = errD_real + errD_fake + errD_fake1
        D.zero_grad()
        errD.backward()
        optimizerD.step()
        # 更新VAE(G)1
        z, mean, logstd = vae.encoder(pgd)
        recon_data = vae.rec_smi(pgd)  # regard z as latent feature and be the robust f
        pre = D_lat(z)
        target = torch.ones_like(pre).to(device)
        loss1 = cri(pre, target)

        vae_loss1 = loss_function(recon_data, data, mean, logstd)
        # + D clear
        output = D(recon_data)
        real_label = torch.ones(batch_size).to(device)
        vae_loss2 = criterion(output, real_label)

        vae.zero_grad()
        vae_loss = 20 * vae_loss1 + vae_loss2 + 10 * loss1
        vae_loss.backward()
        optimizerVAE.step()
        if i % 100 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f  Loss_G: %.4f'
                  % (epoch, nepoch, i, len(train_loader),
                     errD.item(), vae_loss.item()))
        if epoch > 10:
            if i == len(train_loader) - 1:
                real_images = make_grid(pgd.cpu(), nrow=8, normalize=True).detach()
                save_image(real_images, './img_CVAE-GAN/real_images-{}.png'.format(epoch))
                output = vae.rec_smi(pgd)
                fake_images = make_grid(output.cpu(), nrow=8, normalize=True).detach()
                save_image(fake_images, './img_CVAE-GAN/fake_images-{}.png'.format(epoch))
        output = vae.rec_smi(pgd)
        _, pre = mode(output).max(1)
        count += pre.eq(label).sum().item()
        total += label.size(0)
    print(count, total, count / total)


def test(epoch):
    count1 = 0
    total1 = 0
    vae.eval()
    D.eval()
    global best_acc
    with torch.no_grad():
        for i, (pgd, data, label) in enumerate(test_loader):
            pgd = pgd.to(device)
            data = data.to(device)
            label = label.to(device)
            output = vae.rec_smi(pgd)
            _, pre = mode(output).max(1)
            count1 += pre.eq(label).sum().item()
            total1 += label.size(0)
            if epoch > 10:
                if i == len(test_loader) - 1:
                    real_images = make_grid(pgd.cpu(), nrow=8, normalize=True).detach()
                    save_image(real_images, './test_result/real_images-{}.png'.format(epoch))
                    output = vae.rec_smi(pgd)
                    fake_images = make_grid(output.cpu(), nrow=8, normalize=True).detach()
                    save_image(fake_images, './test_result/fake_images-{}.png'.format(epoch))
    acc = count1 / total1
    print("test_acc", acc, count1, total1)
    if acc > best_acc:
        best_acc = acc
        print('Saving..')  # 98.32
        torch.save(vae.state_dict(), './mode_save/CVAE-GAN-VAE.pth')
        torch.save(D.state_dict(), './mode_save/CVAE-GAN-Discriminator.pth')


for epoch in range(0, 2000):
    train(epoch)
    test(epoch)
print(best_acc)
