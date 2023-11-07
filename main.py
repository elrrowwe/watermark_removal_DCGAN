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
print(device)  #making sure the code is run on GPU

#getting train file paths
paths_watermarked, paths_non_watermarked, train_wmark_path, train_nowmark_path = get_paths(split='train')
paths_watermarked_sorted, paths_non_watermarked_sorted = match_file_names(paths_watermarked, paths_non_watermarked, train_wmark_path, train_nowmark_path)

#getting valid file paths
valid_paths_watermarked, valid_paths_non_watermarked, valid_wmark_path, valid_nowmark_path = get_paths(split='val')
valid_paths_watermarked_sorted, valid_paths_non_watermarked_sorted = match_file_names(valid_paths_watermarked, valid_paths_non_watermarked, valid_wmark_path, valid_nowmark_path)


#OLD TRAIN, VALIDATION SPLITS

# paths_watermarked_sorted = process_dataset(paths_watermarked_sorted)
# paths_non_watermarked_sorted = process_dataset(paths_non_watermarked_sorted)
# X_train, X_test, y_train, y_test = train_test_split(paths_watermarked_sorted, paths_non_watermarked_sorted, train_size=0.8, random_state=1, shuffle=True)

wm_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize([int(64), int(64)]),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    ])

nwm_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize([int(64), int(64)]),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    ])

#creating dataloaders for training, validation data
ds_train = WatermarkRemovalData(paths_watermarked_sorted, paths_non_watermarked_sorted, transform=wm_transform, target_transform=nwm_transform)
ds_valid = WatermarkRemovalData(valid_paths_watermarked_sorted, valid_paths_non_watermarked_sorted, transform=wm_transform, target_transform=nwm_transform)

dataloader_train = DataLoader(dataset=ds_train, batch_size=1, shuffle=False)
dataloader_valid = DataLoader(dataset=ds_valid, batch_size=1, shuffle=False)

real_label = 1
fake_label = 0

# creating a Binary Cross-Entropy optimization criterion (loss function, essentially log likelihood for binary classification)
adversarial_loss = nn.BCELoss()

# Lists to keep track of progress
img_list = []
G_losses = []
D_losses = []
iters = 0
num_epochs = 1000

fixed_im_transform = v2.Compose([
        v2.ToImage(),
        v2.Resize([int(64), int(64)]),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0, 0, 0], std=[1, 1, 1])
    ])
fixed_im = 0
netG = Generator().to(device)
netD = Discriminator().to(device)

optimizerG = Adam(netG.parameters(), lr=1e-5, betas=[0.9, 0.9])
optimizerD = Adam(netD.parameters(), lr=1e-5, betas=[0.9, 0.9])
adversarial_loss = nn.BCELoss()

print('===...TRAINING...===')
print(f'len of dataloader = {len(dataloader_train)}')

# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader_train, 0):
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
        label = label.view(-1).expand_as(output)
        # Calculate loss on all-real batch
        errD_real = adversarial_loss(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate fake image batch with G
        fake = netG(data[0].to(device))
        label = torch.full((b_size,), fake_label, dtype=torch.float, device=device)
        # Classify all fake batch with D
        output = netD(fake.detach()).view(-1)
        #match the sizes of label and 
        label = label.view(-1).expand_as(output)
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
                  % (epoch, num_epochs, i, len(dataloader_train),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on a fixed image
        # if (iters % 500 == 0) or ((epoch == num_epochs - 1) and (i == len(dataloader) - 1)):
        #     with torch.no_grad():
        #         fake = netG(fixed_im.to(device)).detach().cpu()
        #     img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1
netG.remove_watermark(dataloader_valid[0])
