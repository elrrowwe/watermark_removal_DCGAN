import torch
import torch.nn as nn
from utils import *
from torch.optim.adam import Adam
from discriminator import Discriminator
from generator import Generator
from custom_dataset import WatermarkRemovalData
from sklearn.model_selection import train_test_split
import torchvision.utils as vutils
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from torchvision.io import read_image

# TODO: read about convolutional autoencoders
# TODO: read about SGD

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)  # Making sure the code is run on GPU

# Lists of filenames
paths_watermarked, paths_non_watermarked, train_wmark_path, train_nowmark_path = get_paths(split='train')
paths_watermarked_sorted, paths_non_watermarked_sorted = match_file_names(paths_watermarked, paths_non_watermarked, train_wmark_path, train_nowmark_path)

# paths_watermarked_sorted = process_dataset(paths_watermarked_sorted)
# paths_non_watermarked_sorted = process_dataset(paths_non_watermarked_sorted)
# X_train, X_test, y_train, y_test = train_test_split(paths_watermarked_sorted, paths_non_watermarked_sorted, train_size=0.8, random_state=1, shuffle=True)

wm_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize([int(128), int(128)]),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    ])

nwm_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize([int(128), int(128)]),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    ])

ds = WatermarkRemovalData(paths_watermarked_sorted, paths_non_watermarked_sorted, transform=wm_transform, target_transform=nwm_transform)

dataloader = DataLoader(dataset=ds, batch_size=1, shuffle=False)

real_label = 1
fake_label = 0

# Training
adversarial_loss = nn.BCELoss()

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0
num_epochs = 1000

fixed_im_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize([int(128), int(128)]),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    ])
fixed_im = fixed_im_transform(read_image('C:\\Users\\death\\Desktop\\rnns\\valid\\watermark\\eyes-cats-cat-couch.jpg'))

netG = Generator().to(device)
netD = Discriminator().to(device)

optimizerG = Adam(netG.parameters(), lr=1e-4)
optimizerD = Adam(netD.parameters(), lr=1e-4)
adversarial_loss = nn.BCELoss()

print('===...TRAINING...===')
print(f'len of dataloader = {len(dataloader)}')

# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        no_watermarks_gpu = data[1].to(device)
        # transform = T.ToPILImage()
        # img = transform(no_watermarks_gpu.cpu())
        # img.show()
        b_size = data[0].size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        output = netD(no_watermarks_gpu)
        label = label.view(b_size, 1, 1, 1).expand_as(output)
        # Calculate loss on all-real batch
        errD_real = adversarial_loss(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate fake image batch with G
        fake = netG(data[0].to(device))
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = adversarial_loss(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output = netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG = adversarial_loss(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
            with torch.no_grad():
                fake = netG(fixed_im.to(device)).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
