from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch

class MnistDataset(Dataset):
    def __init__(self, images, labels, transforms):
        super().__init__()
        self.images = images
        self.labels = labels
        self.transforms = transforms
        
    
    def __getitem__(self, index):
        image = self.images[index, :, :]
        label = self.labels[index]
        image = image[None, :, :]
        image = self.transforms(image)
        
        # return image, label # for classification task
        return image, image  # For image generation task
    
    
    def __len__(self):
        return self.images.shape[0]
    

def custom_worker_init_func(x):
    return np.random.seed((torch.initial_seed()) % (2**32))

def get_loader(dataset, batchsize, num_workers, shuffle=True, drop_last=False):
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batchsize,
                            num_workers=num_workers,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            worker_init_fn=custom_worker_init_func)
    
    return dataloader