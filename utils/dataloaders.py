from torch.utils.data import Dataset, DataLoader


class MnistDataset(Dataset):
    def __init__(self, images, labels, transforms):
        super().__init__()
        self.images = images
        self.labels = labels
        self.transforms = transforms
        
    
    def __getitem__(self, index):
        pass
    
    
    def __len__(self, index):
        pass