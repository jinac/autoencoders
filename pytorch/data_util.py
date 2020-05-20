"""
data utility to load datasets.
"""

import torchvision
import torch

class AnimeFaceData(object):
	def __init__(self, img_dim=[64, 64], batch_size=32, shuffle=True, num_workers=4):
		self.path = '/media/jinc/shared_sys/anime_faces/'
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.img_dim = img_dim

		self.transform = torchvision.transforms.Compose([
			torchvision.transforms.RandomHorizontalFlip(),
			torchvision.transforms.RandomRotation(45),
			torchvision.transforms.Resize(img_dim),
			torchvision.transforms.ToTensor(),])
		self.img_folder = torchvision.datasets.ImageFolder(self.path, transform=self.transform)
		self.data_loader = torch.utils.data.DataLoader(self.img_folder,
													   batch_size=self.batch_size,
													   shuffle=shuffle,
													   num_workers=self.num_workers)
		print(self.img_folder)