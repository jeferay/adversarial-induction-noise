import torch
import torchvision
from torch.utils.data import DataLoader
from dataset import PoisonWebFace10

import random
import matplotlib.pyplot as plt
import matplotlib

import numpy as np
train_transform = [
    torchvision.transforms.RandomCrop(32, padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor()
]
train_transform = torchvision.transforms.Compose(train_transform)




def get_unlearnable_train_dataset(perturb_tensor_filepath='../experiments/mada/samplewise/cifar10/at_from_scratch_epsilon32_plus_one_batch128_rs_stepsize.8_trainstep30/perturbation/resnet18_madrys/perturbation.pt'):

    noise= torch.load(perturb_tensor_filepath)
    clean_train_dataset = torchvision.datasets.CIFAR10(root='../datasets', train=True, download=True, transform=train_transform)
    print(clean_train_dataset.data.shape)
    unlearnable_train_dataset = torchvision.datasets.CIFAR10(root='../datasets', train=True, download=True, transform=train_transform)
    perturb_noise = noise.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()#shape of n,32,32,3:这里的范围是0-255
    print(perturb_noise.shape)
    unlearnable_train_dataset.data = unlearnable_train_dataset.data.astype(np.float32)
    for i in range(len(unlearnable_train_dataset)):
        unlearnable_train_dataset.data[i] += perturb_noise[i]
        unlearnable_train_dataset.data[i] = np.clip(unlearnable_train_dataset.data[i], a_min=0, a_max=255)
    unlearnable_train_dataset.data = unlearnable_train_dataset.data.astype(np.uint8)#数据范围是0~255

    return clean_train_dataset, unlearnable_train_dataset,noise

def img_show_face(perturb_tensor_filepath='../experiments/mada/samplewise/webface-10/at_pretrain_epsilon32/perturbation/resnet18_madrys/perturbation.pt'):
    poison_train_dataset = PoisonWebFace10(root='../datasets/webface-10-train', transform=torchvision.transforms.Compose([
                            torchvision.transforms.Resize(size=(224,224)),
                            torchvision.transforms.ToTensor()]),pertub_tensor_filepath='../experiments/mada/samplewise/webface-10/at_pretrain_epsilon32/perturbation/resnet18_madrys/perturbation.pt')
    
    clean_train_dataset = torchvision.datasets.ImageFolder(root='../datasets/webface-10-train', transform=torchvision.transforms.Compose([
                            torchvision.transforms.Resize(size=(224,224)),
                            torchvision.transforms.ToTensor()]))
    img_grid=[]
    for seed in range(5):
        random.seed(seed)
        selected_idx = selected_idx = random.randint(0, 4000)
        poison_data = poison_train_dataset.__getitem__(selected_idx)[0]
        clean_data  = clean_train_dataset.__getitem__(selected_idx)[0]
        poison_data = torch.clamp(poison_data, 0, 1)

        img_grid.append(poison_data)
        img_grid.append(clean_data)
    
    
    npimg = torchvision.utils.make_grid(torch.stack(img_grid), nrow=2, pad_value=255).numpy()
    fig = plt.figure(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #print(npimg.shape)
    plt.show()
    plt.savefig('visual_pretain_face.png')

    

    
    


def imshow(img):
    fig = plt.figure(figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    print(npimg.shape)
    plt.show()
    plt.savefig('visual_pretain_face.png')
    
    
    
    
def get_pairs_of_imgs(idx,clean_train_dataset,unlearnable_train_dataset,noise):
    clean_img = clean_train_dataset.data[idx]
    unlearnable_img = unlearnable_train_dataset.data[idx]
    clean_img = torchvision.transforms.functional.to_tensor(clean_img)# to tensor就是放缩到0，1区间的
    unlearnable_img = torchvision.transforms.functional.to_tensor(unlearnable_img)

    x = noise[idx]
    x_min = torch.min(x)
    x_max = torch.max(x)
    noise_norm = (x - x_min) / (x_max - x_min)
    noise_norm = torch.clamp(noise_norm, 0, 1)
    return [clean_img, noise_norm, unlearnable_img]

# random.seed(0)    
# selected_idx = random.randint(0, 50000)
# img_grid = []

# clean_train_dataset,unlearnable_train_dataset,noise = get_unlearnable_train_dataset()
# img_grid += get_pairs_of_imgs(selected_idx,clean_train_dataset,unlearnable_train_dataset,noise)



# imshow(torchvision.utils.make_grid(torch.stack(img_grid), nrow=1, pad_value=255))

img_show_face()