import os
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from PIL import Image
import matplotlib.pyplot as plt


DATASET_PATH = "./data/images"
OUTPUT_DIR = "./outputs"
MODEL_DIR = "./models"

IMAGE_SIZE = 32
CHANNELS = 3
LATENT_DIM = 100
BATCH_SIZE = 4
LEARNING_RATE = 0.0002
BETA1 = 0.5
EPOCHS = 10

NGF = 64
NDF = 64

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATASET_PATH, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cpu":
    print("WARNING: No CUDA GPU detected - training will run on CPU and will be slow. "
          "Consider using the provided Colab notebook (colab_train.ipynb) for free GPU training, "
          "or reduce IMAGE_SIZE/BATCH_SIZE/EPOCHS below for a quick local test run.")


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_paths = []
        for ext in SUPPORTED_EXTENSIONS:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, f"*{ext}")))
            self.image_paths.extend(glob.glob(os.path.join(root_dir, f"*{ext.upper()}")))
        self.image_paths = sorted(set(self.image_paths))

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No images found in '{root_dir}'. "
                f"Please add .jpg/.jpeg/.png files there before running the script."
            )
        print(f"Found {len(self.image_paths)} training images in '{root_dir}'.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

dataset = CustomImageDataset(DATASET_PATH, transform=transform)

if len(dataset) < BATCH_SIZE:
    raise RuntimeError(
        f"BATCH_SIZE={BATCH_SIZE} is larger than the dataset ({len(dataset)} images). "
        f"With drop_last=True this produces zero batches, so the training loop would "
        f"silently do nothing. Add more images to '{DATASET_PATH}' or lower BATCH_SIZE "
        f"to at most {len(dataset)}."
    )

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=0,
)

if len(dataloader) == 0:
    raise RuntimeError(
        f"BATCH_SIZE={BATCH_SIZE} combined with drop_last=True produces zero batches "
        f"from a dataset of {len(dataset)} images. Add more images or lower BATCH_SIZE."
    )


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, nz=LATENT_DIM, ngf=NGF, nc=CHANNELS):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 4, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self, nc=CHANNELS, ndf=NDF):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(nc, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, img):
        out = self.model(img)
        return out.view(-1, 1).squeeze(1)


generator = Generator().to(device)
discriminator = Discriminator().to(device)
generator.apply(weights_init)
discriminator.apply(weights_init)

adversarial_loss = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))

fixed_noise = torch.randn(16, LATENT_DIM, device=device)


def show_and_save_images(images, epoch):
    grid = make_grid(images, nrow=4, normalize=True)

    plt.figure(figsize=(4, 4))
    plt.imshow(grid.permute(1, 2, 0).cpu().detach())
    plt.title(f"Generated Images - Epoch {epoch}")
    plt.axis("off")
    plt.show(block=False)
    plt.pause(0.1)
    plt.close()

    save_path = os.path.join(OUTPUT_DIR, f"epoch_{epoch:04d}.png")
    save_image(images, save_path, nrow=4, normalize=True)
    print(f"Saved sample grid to {save_path}")


for epoch in range(1, EPOCHS + 1):
    for i, real_imgs in enumerate(dataloader):
        real_imgs = real_imgs.to(device)
        batch_size_curr = real_imgs.size(0)

        valid = torch.ones(batch_size_curr, device=device)
        fake = torch.zeros(batch_size_curr, device=device)

        optimizer_G.zero_grad()
        z = torch.randn(batch_size_curr, LATENT_DIM, device=device)
        gen_imgs = generator(z)
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        print(f"[Epoch {epoch}/{EPOCHS}] [Batch {i}/{len(dataloader)}] "
              f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

    generator.eval()
    with torch.no_grad():
        sample_imgs = generator(fixed_noise)
    generator.train()

    show_and_save_images(sample_imgs, epoch)

torch.save(generator.state_dict(), os.path.join(MODEL_DIR, "generator.pth"))
torch.save(discriminator.state_dict(), os.path.join(MODEL_DIR, "discriminator.pth"))
print(f"Saved trained models to '{MODEL_DIR}/generator.pth' and '{MODEL_DIR}/discriminator.pth'")

print("Training finished!")