import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib 
#matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
#from skimage.metrics import structural_similarity as ssim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, RandomSampler 
from PIL import Image

def save_fake_img(img, folder_path):
    
    batch_size = img.size()[0]
    for idx in range(batch_size):
        indv_img = transforms.ToPILImage()(img[idx,:,:,:])
        file_path = os.path.join(folder_path, (str(idx)+'.png'))
        indv_img.save(file_path)
        #print(f"Image {idx} has been saved in {file_path}")
    print(f"Fake mages have been saved in {folder_path}")

###### MODIFIED FROM DCGAN TUTORIAL OBTAINED FROM PYTORCH ######      

def weights_init(m):
    """
    # custom weights initialization called on netG and netD

    Model weights should be randomly initialised from normal distribution with mean = 0, stdev = 0.02
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

class Generator(nn.Module):

    def __init__(self, SIZE_Z = 100, NUM_GEN_FEATURES = 64):
        """
        Maps latent vector (z) to data space (data image with same size as training images)
        
        Consists of multiple conv2d
        
        """
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( SIZE_Z, NUM_GEN_FEATURES * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NUM_GEN_FEATURES * 8),
            nn.ReLU(True),
            
            # state size. (NUM_GEN_FEATURES*8) x 4 x 4
            nn.ConvTranspose2d(NUM_GEN_FEATURES * 8, NUM_GEN_FEATURES * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NUM_GEN_FEATURES * 4),
            nn.ReLU(True),
            # state size. (NUM_GEN_FEATURES*4) x 8 x 8
            nn.ConvTranspose2d( NUM_GEN_FEATURES * 4, NUM_GEN_FEATURES * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NUM_GEN_FEATURES * 2),
            nn.ReLU(True),
            # state size. (NUM_GEN_FEATURES*2) x 16 x 16
            nn.ConvTranspose2d( NUM_GEN_FEATURES * 2, NUM_GEN_FEATURES, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NUM_GEN_FEATURES),
            nn.ReLU(True),
            # state size. (NUM_GEN_FEATURES) x 32 x 32
            nn.ConvTranspose2d(NUM_GEN_FEATURES, NUM_CHANNELS, 4, 2, 1, bias=False),
            #nn.Upsample((100,100,24), mode = 'trilinear'),
            nn.Tanh() # maps input features to between -1 to 1
            # state size. (nc) x 64 x 64
        )
        
    def forward(self, input):
        return self.main(input)
    
class Discriminator(nn.Module):
    """
    Input : Image : 3x64x64 
    Output : probability that image is real

    SIGMOID is applied to obtain probability that image is real or not 
    
    """
    def __init__(self, NUM_CHANNELS =2, NUM_DIS_FEATURES = 64):    # Size of feature maps in discriminator)
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            # input is (NUM_CHANNELS) x 64 x 64
            nn.Conv2d(NUM_CHANNELS, NUM_DIS_FEATURES, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (NUM_DIS_FEATURES) x 32 x 32
            nn.Conv2d(NUM_DIS_FEATURES, NUM_DIS_FEATURES * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NUM_DIS_FEATURES * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (NUM_DIS_FEATURES*2) x 16 x 16
            nn.Conv2d(NUM_DIS_FEATURES * 2, NUM_DIS_FEATURES * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NUM_DIS_FEATURES * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # state size. (NUM_DIS_FEATURES*4) x 8 x 8
            nn.Conv2d(NUM_DIS_FEATURES * 4, NUM_DIS_FEATURES * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NUM_DIS_FEATURES * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.final_conv = nn.Conv2d(NUM_DIS_FEATURES * 8, 1, 4, 1, 0, bias=False)
        #self.flatten = nn.Flatten()
        #self.linear_layer = nn.Linear(NUM_DIS_FEATURES*13*13, 1)
        self.sigmoid_output = nn.Sigmoid() # SQUASH output between -1,1
        
    def forward(self, input):
        out = self.main(input)
        out = self.final_conv(out)
        #out = self.linear_layer(out)
        #out = self.flatten(out)
        out = self.sigmoid_output(out)
        return out

class PokeDataset(Dataset):
    """
    Defines pokemon dataset 
    """
    
    def __init__(self, folder_name, resize_size = 256, use_transform = True):
        
        self.folder_name = folder_name
        all_images = os.listdir(folder_name)
        self.all_image_path = [os.path.join(folder_name,file) for file in all_images if ".DS_Store" not in file]
        self.data_len = len(self.all_image_path)
        self.transform = self.apply_transform()
        self.resize_size = resize_size
    
    def __len__(self):
        return self.data_len
    
    def _normalise(self, img):
        """
        A function that converts an image volume from np.float64 to uint8 
        between 0 to 255

        """
        max_img = np.max(img)
        min_img = np.min(img)

        #Normalise values between 0 to 1
        normalised_img =  ((img - min_img)/(max_img - min_img)) 

        return normalised_img.astype(np.float32)

    def __getitem__(self, item):
        
        image = Image.open(self.all_image_path[item])
        image = image.resize((self.resize_size, self.resize_size), resample=Image.BILINEAR)
        image = self.transform(image)
        
        return image
      
    def apply_transform(self):     
        transform =   transforms.Compose([
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))
                                ])
        return transform

if __name__ == '__main__':
    
    # DEFINING HYPERPARAMETERS / PARAMETERS
    BATCH_SIZE = 32
    NUM_CHANNELS = 4  # Number of channels in the training images. Using 3d volume -> Channels = z_dim = 48 
    SIZE_Z = 100    # Size of z latent vector (i.e. size of generator input)
    NUM_GEN_FEATURES = 64 # Size of feature maps in generator
    NUM_DIS_FEATURES = 64    # Size of feature maps in discriminator
    NUM_EPOCHS = 100  # Number of training epochs
    LR = 0.0001
    BETA1 = 0.5 # Beta1 hyperparam for Adam optimizers
    NUM_GPU = 1    # Number of GPUs available. Use 0 for CPU mode.
    device = torch.device("cuda" if (torch.cuda.is_available() and NUM_GPU > 0) else "cpu")
    
    # DATASET FOLDER 
    folder = './data'
    poke_ds = PokeDataset(folder, resize_size=64)
    poke_dataloader = DataLoader(poke_ds, batch_size = BATCH_SIZE)
    BASE_DIR = os.path.join(os.getcwd(), 'RESULTS') # for saving results into 
    os.makedirs(BASE_DIR, exist_ok=True)
    
    Generator_net = Generator().to(device)
    Generator_net.apply(weights_init)  # Apply the weights_init function to randomly initialize all weightsto mean=0, stdev=0.2.
    Discriminator_net = Discriminator(NUM_CHANNELS=NUM_CHANNELS, NUM_DIS_FEATURES=64).to(device)
    Discriminator_net.apply(weights_init)
    
    # Defining loss function
    criterion = nn.BCELoss()
    
    # Create latent space vectors for generator 
    fixed_noise = torch.randn(NUM_GEN_FEATURES, SIZE_Z, 1, 1, device=device)
    
    real_label = 1.
    fake_label = 0.
    optimizerD = torch.optim.Adam(Discriminator_net.parameters(), lr=LR, betas=(BETA1, 0.999))
    optimizerG = torch.optim.Adam(Generator_net.parameters(), lr=LR, betas=(BETA1, 0.999))


    G_losses = []
    D_losses = []
    iters = 0
    
    # For saving h5py file and summary staistics 
    RUNS_DIR = os.path.join(BASE_DIR, 'runs')
    os.makedirs(RUNS_DIR, exist_ok=True)
    writer = SummaryWriter(RUNS_DIR)    

    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(NUM_EPOCHS):
        
        # For each batch in the dataloader
        for i, data in enumerate(poke_dataloader):
           
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            
            ## Train with all-real batch
            Discriminator_net.zero_grad()
            
            # Format batch
            #real_cpu = data[0].to(device)
            data = data.to(device)
            b_size = data.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
  
            # Forward pass real batch through D
            output = Discriminator_net(data).view(-1)
            
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, SIZE_Z, 1, 1, device=device)
            
            # Generate fake image batch with G
            fake = Generator_net(noise)
            label.fill_(fake_label)

            # Classify all fake batch with D
            output = Discriminator_net(fake.detach()).view(-1)

            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
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
            Generator_net.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = Discriminator_net(fake).view(-1)
            
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            # Update G
            optimizerG.step()
            
            # Output training stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, NUM_EPOCHS, i, len(poke_dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                #print(f'SSIM metric: {ssm_metric:.2f}')

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            
            # Save summary statistics
            writer.add_scalar('Loss/Generator', errG.item(), iters)
            writer.add_scalar('Loss/Discrimantor', errD.item(), iters)
            #writer.add_scalar('Metrics/SSIM', ssm_metric, iters)

            # Check how the generator is doing by saving G's output on latent_space_vector
            if (iters % 100 == 0) or ((epoch == NUM_EPOCHS -1) and (i == len(poke_dataloader)-1)):
                with torch.no_grad():
                    fake =  Generator_net(fixed_noise).detach().cpu()

                # Save fake images to folder under base_dir 
                ITER_FOLD = os.path.join(BASE_DIR, str(iters))
                os.makedirs(ITER_FOLD, exist_ok=True)
                save_fake_img(fake, ITER_FOLD)
                
                # Save models after every 100 steps 
                gen_model_path = os.path.join(BASE_DIR, (str(iters) + '_'+'gen_model.pth'))
                dis_model_path = os.path.join(BASE_DIR, (str(iters) + '_'+'dis_model.pth'))
                torch.save(Discriminator_net.state_dict(), dis_model_path)
                torch.save(Generator_net.state_dict(), gen_model_path)

            iters += 1

    # Save final models:
    gen_model_path = os.path.join(BASE_DIR, 'final_gen_model.pth')
    dis_model_path = os.path.join(BASE_DIR, 'final_gen_model.pth')
    torch.save(Discriminator_net.state_dict(), dis_model_path)
    torch.save(Generator_net.state_dict(), gen_model_path)
    
    print('Training finished!')
    

            
        
        