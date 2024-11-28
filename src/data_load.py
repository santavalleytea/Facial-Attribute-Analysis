import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

img_folder = '../img_align_celeba'
attr_file = '../text_files/list_attr_celeba.txt'
part_file = '../text_files/list_eval_partition.txt'

# Handle image load, reading labels and partition dataset
class CelebDataset(Dataset):
    def __init__(self, img_folder, attr_file, part_file, transform=None, partition=0, max_images=None, single_image=None):
        self.img_folder = img_folder
        self.labels = pd.read_csv(attr_file, sep='\s+', header=1, index_col=0, on_bad_lines='skip')
        self.part_data = pd.read_csv(part_file, sep='\s+', header=None) # Load partition information
        self.partition = partition # 0 = train, 1 = val, 2 = test
        self.transform = transform

        # For single image check purposes
        if single_image:
            self.img_files = [single_image];
        else:
            self.img_files = self.part_data[self.part_data[1] == self.partition][0].values

        if max_images is not None:
            self.img_files = self.img_files[:max_images]

    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_folder, img_name)
        image = Image.open(img_path)

        label = self.labels.loc[img_name].values.astype(float)
        label = torch.tensor(label, dtype=torch.float32)

        label = torch.where(label == -1, torch.tensor(0.0), label)

        if self.transform:
            image = self.transform(image)

        return image, label

# Define trasnformations for dataset
transform = transforms.Compose ([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
])

# Create datasets for training, validation, and testing
train_dataset = CelebDataset(img_folder, attr_file, part_file, transform=transform, partition=0, max_images=1000)  # training
val_dataset = CelebDataset(img_folder, attr_file, part_file, transform=transform, partition=1, max_images=1000)    # validation
test_dataset = CelebDataset(img_folder, attr_file, part_file, transform=transform, partition=2, max_images=1000)   # testing

# Create DataLoaders to load data in batches
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

if __name__ == '__main__':
    # Testing for a single image
    single_image = '000001.jpg'  
    
    single_image_dataset = CelebDataset(img_folder, attr_file, part_file, transform=transform, partition=0, single_image=single_image)
    
    single_image_loader = DataLoader(single_image_dataset, batch_size=1, shuffle=False)
    
    for images, labels in single_image_loader:
        print(f'Image shape: {images.shape}')
        print(f'Label shape: {labels.shape}')
        break
