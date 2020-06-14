"""
data utility to load datasets.
"""
import torchvision
import torch


class AnimeFaceData(object):
    def __init__(self, img_dim=[64, 64], batch_size=32, shuffle=True, num_workers=4, pin_memory=True):
        self.path = '/media/jinc/shared_sys/anime_faces/'
        # self.path = '/test/data/'
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_dim = img_dim
        self.pin_memory = pin_memory

        # Apply random horizontal flip, 45 deg rotation, 0.1 translation x-y, rescaling.
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.RandomAffine(degrees=45,
            #                                     translate=(0.1, 0.1),
            #                                     scale=(0.75, 1.25),
            #                                     ),
            torchvision.transforms.Resize(img_dim),
            torchvision.transforms.ToTensor()])
        self.img_folder = torchvision.datasets.ImageFolder(self.path, transform=self.transform)
        self.train_len = int(0.8 * len(self.img_folder))
        self.test_len = len(self.img_folder) - self.train_len
        self.train_data, self.test_data = torch.utils.data.random_split(self.img_folder, [self.train_len, self.test_len])
        self.data_loader = torch.utils.data.DataLoader(self.train_data,
                                                       batch_size=self.batch_size,
                                                       shuffle=shuffle,
                                                       num_workers=self.num_workers,
                                                       pin_memory=self.pin_memory)
        self.test_loader = torch.utils.data.DataLoader(self.test_data,
                                                       batch_size=self.batch_size,
                                                       shuffle=False,
                                                       num_workers=self.num_workers,
                                                       pin_memory=self.pin_memory)