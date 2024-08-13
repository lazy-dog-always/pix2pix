import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import time

# 数据集路径
dataset_path = 'C:/Users/admin/Desktop/AI风景model升级版/dataset'
BUFFER_SIZE = 400
BATCH_SIZE = 16
IMG_WIDTH = 256
IMG_HEIGHT = 256
LAMBDA = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义数据集
class ImageDataset(Dataset):
    def __init__(self, blurry_paths, clear_paths, transform=None):
        self.blurry_paths = blurry_paths
        self.clear_paths = clear_paths
        self.transform = transform

    def __len__(self):
        return len(self.blurry_paths)

    def __getitem__(self, idx):
        blurry_image = Image.open(self.blurry_paths[idx]).convert('RGB')
        clear_image = Image.open(self.clear_paths[idx]).convert('RGB')

        if self.transform:
            blurry_image = self.transform(blurry_image)
            clear_image = self.transform(clear_image)

        return blurry_image, clear_image

def get_dataset_paths(dataset_path, phase):
    blurry_images = sorted(os.listdir(os.path.join(dataset_path, phase, 'blurry')))
    clear_images = sorted(os.listdir(os.path.join(dataset_path, phase, 'clear')))
    blurry_paths = [os.path.join(dataset_path, phase, 'blurry', img) for img in blurry_images]
    clear_paths = [os.path.join(dataset_path, phase, 'clear', img) for img in clear_images]
    return blurry_paths, clear_paths

# 数据变换
transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

blurry_train_paths, clear_train_paths = get_dataset_paths(dataset_path, 'train')
blurry_test_paths, clear_test_paths = get_dataset_paths(dataset_path, 'test')

train_dataset = ImageDataset(blurry_train_paths, clear_train_paths, transform=transform)
test_dataset = ImageDataset(blurry_test_paths, clear_test_paths, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

# 构建Pix2Pix模型
class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, size, apply_batchnorm=True):
        super(Downsample, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=size, stride=2, padding=1, bias=False)
        ]
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(inplace=False))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, size, apply_dropout=False):
        super(Upsample, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=size, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ]
        if apply_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=False))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            Downsample(6, 64, 4, apply_batchnorm=False),
            Downsample(64, 128, 4),
            Downsample(128, 256, 4),
            nn.ZeroPad2d(1),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=False),
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1)
        )

    def forward(self, x):
        return self.model(x)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.down_stack = nn.ModuleList([
            Downsample(3, 64, 4, apply_batchnorm=False),
            Downsample(64, 128, 4),
            Downsample(128, 256, 4),
            Downsample(256, 512, 4),
            Downsample(512, 512, 4),
            Downsample(512, 512, 4),
            Downsample(512, 512, 4),
            Downsample(512, 512, 4)
        ])
        self.up_stack = nn.ModuleList([
            Upsample(512, 512, 4, apply_dropout=True),
            Upsample(1024, 512, 4, apply_dropout=True),
            Upsample(1024, 512, 4, apply_dropout=True),
            Upsample(1024, 512, 4),
            Upsample(1024, 256, 4),
            Upsample(512, 128, 4),
            Upsample(256, 64, 4)
        ])
        self.last = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        skips = []
        for down in self.down_stack:
            x = down(x)
            skips.append(x)
        skips = reversed(skips[:-1])
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
        x = self.last(x)
        return x

# 初始化模型
generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

# 定义损失函数
criterion = nn.BCEWithLogitsLoss()

# 定义优化器
generator_optimizer = optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))

# 定义训练步骤
def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = criterion(disc_generated_output, torch.ones_like(disc_generated_output, device=DEVICE))
    l1_loss = torch.mean(torch.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = criterion(disc_real_output, torch.ones_like(disc_real_output, device=DEVICE))
    generated_loss = criterion(disc_generated_output, torch.zeros_like(disc_generated_output, device=DEVICE))
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

# 训练和测试步骤
def train_step(input_image, target):
    generator_optimizer.zero_grad()
    discriminator_optimizer.zero_grad()

    # 生成图像
    gen_output = generator(input_image)

    # 判别器输出
    disc_real_output = discriminator(torch.cat([input_image, target], dim=1))
    disc_generated_output = discriminator(torch.cat([input_image, gen_output], dim=1))

    # 计算生成器损失并反向传播
    gen_loss = generator_loss(disc_generated_output, gen_output, target)
    gen_loss.backward()  # 生成器反向传播
    generator_optimizer.step()

    # 计算判别器损失并反向传播
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
    disc_loss.backward()  # 判别器反向传播
    discriminator_optimizer.step()

    return gen_loss.item(), disc_loss.item()

def compute_metrics(loader):
    total_gen_loss = 0.0
    total_disc_loss = 0.0
    num_batches = 0

    for input_image, target in loader:
        input_image = input_image.to(DEVICE)
        target = target.to(DEVICE)
        gen_loss, disc_loss = train_step(input_image, target)
        total_gen_loss += gen_loss
        total_disc_loss += disc_loss
        num_batches += 1

    return total_gen_loss / num_batches, total_disc_loss / num_batches

def fit(train_loader, epochs, test_loader):
    torch.autograd.set_detect_anomaly(True)  # 启用异常检测
    for epoch in range(epochs):
        start = time.time()
        print(f"Starting Epoch {epoch + 1}/{epochs}")

        train_gen_loss, train_disc_loss = compute_metrics(train_loader)
        test_gen_loss, test_disc_loss = compute_metrics(test_loader)

        print(f"Epoch {epoch + 1} Training Gen Loss: {train_gen_loss:.4f}, Disc Loss: {train_disc_loss:.4f}")
        print(f"Epoch {epoch + 1} Test Gen Loss: {test_gen_loss:.4f}, Disc Loss: {test_disc_loss:.4f}")
        print(f"Completed Epoch {epoch + 1} in {time.time() - start:.2f} seconds")

    torch.save(generator.state_dict(), 'C:/Users/admin/Desktop/AI风景model升级版/model-gen/generator2.pth')
    torch.save(discriminator.state_dict(), 'C:/Users/admin/Desktop/AI风景model升级版/model-dis/discriminator2.pth')

if __name__ == '__main__':
    # 开始训练
    fit(train_loader, epochs=1, test_loader=test_loader)
