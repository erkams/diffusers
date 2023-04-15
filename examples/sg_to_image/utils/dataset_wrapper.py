from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch

class HFDataset(Dataset):
    def __init__(self, dset, image_key='target_img', length=None):
        self.dset = dset
        self.image_key = image_key
        if length is not None:
            self.length = length
        else:
            self.length = len(dset)

    def __getitem__(self, idx):
        img = self.dset[idx][self.image_key]
        # if img is PIL image, convert to torch tensor
        if isinstance(img, Image.Image):
            img = transforms.ToTensor()(img)
            # to torch.uint8
            img = (255*img).to(torch.uint8)
        elif isinstance(img, torch.Tensor):
            if img.dtype != torch.uint8:
                raise ValueError('Image tensor must be of type torch.uint8')
            
        return img

    def __len__(self):
        return self.length

class PredictionSet(Dataset):
    """
    Turn the list of images in torch tensor type into a torch dataset
    """
    def __init__(self, imgs):
        self.imgs = imgs
    
    def __getitem__(self, idx):
        if isinstance(self.imgs[idx], Image.Image):
            self.imgs[idx] = transforms.ToTensor()(self.imgs[idx])
            # to torch.uint8
            self.imgs[idx] = (255*self.imgs[idx]).to(torch.uint8)
        elif isinstance(self.imgs[idx], torch.Tensor):
            if self.imgs[idx].dtype != torch.uint8:
                raise ValueError('Image tensor must be of type torch.uint8')
        return self.imgs[idx]
    
    def __len__(self):
        return len(self.imgs)
    
class ListDataset(Dataset):
    """
    Turn the list of images in torch tensor type into a torch dataset
    """
    def __init__(self, imgs):
        self.imgs = imgs
    
    def __getitem__(self, idx):
        # if img is PIL image, convert to torch uint8 tensor
        if isinstance(self.imgs[idx], Image.Image):
            self.imgs[idx] = transforms.ToTensor()(self.imgs[idx])
            # to torch.uint8
            self.imgs[idx] = (255*self.imgs[idx]).to(torch.uint8)
        elif isinstance(self.imgs[idx], torch.Tensor):
            if self.imgs[idx].dtype != torch.uint8:
                raise ValueError('Image tensor must be of type torch.uint8')
        return self.imgs[idx]
    
    def __len__(self):
        return len(self.imgs)