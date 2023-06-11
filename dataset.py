from torch.utils.data import Dataset
from torchvision import transforms
from glob import glob
from PIL import Image
import os

class ImageClassificationDataset(Dataset):
    
	def __init__(self, dataset_dir, height=98, 
	      		 width=50, transformations=None):
		
		self.class_to_idx = {
			'background': 0,
			'down': 1,
			'go': 2,
			'left': 3,
			'no': 4,
			'off': 5,
			'on': 6,
			'right': 7,
			'stop': 8,
			'unknown': 9,
			'up': 10,
			'yes': 11
		}
		self.idx_to_class = {value: key for key, value in self.class_to_idx.items()}
		self.dataset_dir = os.path.join(dataset_dir, '*', '*.png')
		self.height = height
		self.width = width
		if transformations is not None:
			self.transformations = transformations
		else:
			self.transformations = transforms.ToTensor()
		self.load_dataset()

	def load_dataset(self):

		self.images = glob(self.dataset_dir)
		self.labels = [self.class_to_idx[os.path.basename(os.path.dirname(i))] for i in self.images]

	def __len__(self):

		return len(self.labels)
	
	def __getitem__(self, index):

		image = Image.open(self.images[index]).convert('L')
		label = self.labels[index]

		image = self.transformations(image)

		return {
			'image': image,
			'label': label
		}


class ImageGenerationDataset(Dataset):
    
	def __init__(self, dataset_dir, height=98, 
	      		 width=50, transformations=None):
		
		self.dataset_dir = dataset_dir
		self.height = height
		self.width = width
		self.transforms = transformations

	def load_dataset(self):

		self.images = glob(self.dataset_dir)

	def __len__(self):

		return len(self.labels)
	
	def __getitem__(self, index):

		image = Image.open(self.images[index])

		if self.transforms:
			image = self.transforms(image)

		return {
			'image': image
		}