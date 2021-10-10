import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import DatasetFolder, ImageFolder
from torch.utils.data import ConcatDataset, DataLoader, Subset
import torch.nn as nn
import matplotlib.pyplot as plt
from argparse import ArgumentParser, Namespace
from PIL import Image
from tqdm import trange
from tqdm.auto import tqdm
from classifier import *
from itertools import compress
from pathlib import Path
import sys

CNN = ['classifier', 'resnet18', 'resnet50']
REGRESSION = ['logistic', 'linear']

def main(args):

	"""Define the transformation we wish to apply.In the transformation, we also did data augmentation 
	by adding guassian blur and random rotation. Finally, we normalize it to the format of Resnet """
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
								 std=[0.229, 0.224, 0.225])

	train_tfm = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.GaussianBlur(kernel_size=(7, 7), sigma=(0.1, 5)),
		transforms.RandomRotation(degrees=(0, 180)),
		transforms.ToTensor(),
		normalize
	])

	test_tfm = transforms.Compose([
		transforms.Resize((224, 224)),
		transforms.ToTensor(),
		normalize
	])

	#Create the dataset using ImageFolder, which also applies the transformation defined above
	train_set = ImageFolder(args.data_path + 'TRAIN', transform=train_tfm)
	eval_set = ImageFolder(args.data_path + 'TEST', transform=test_tfm)

	#Create Dataloader object so that our data is iterable and ready to train
	trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
	evalloader = DataLoader(eval_set, batch_size=1, shuffle=False)

	#Specify whether we are training on GPU or CPU
	device = "cuda" if torch.cuda.is_available() else "cpu"

	#create model
	torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
	if args.model == 'classifier':
		model = Classifier().to(device)
	elif args.model == 'resnet18' or args.model == 'resnet50':
		if args.model == 'resnet18':
			model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True) 
		else:
			model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True) 
		num_ftrs = model.fc.in_features
		#Append linear layers to the last layer of ResNet to do transfer learning
		model.fc = nn.Sequential(
					nn.Linear(num_ftrs, 128),
					nn.ReLU(),
					nn.Linear(128, 2)).to(device)
		model.to(device)
	elif args.model == 'logistic':
		model = LogisticClassifier().to(device)
	else:
		model = LinearClassifier().to(device)

	#Create loss function
	if args.model in CNN:
		loss_fn = nn.CrossEntropyLoss()
	elif args.model == 'logistic':
		loss_fn =  nn.BCELoss()
	else:
		loss_fn = nn.MSELoss()

	#Create optimizer
	optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

	#Training and evaluating. We also save our model weights once per epoch
	epoch_pbar = trange(args.num_epoch, desc="Epoch")
	for epoch in epoch_pbar:
		train_loop(trainloader, model, loss_fn, optimizer, device, args.model)
		eval_loop(evalloader, model, device, args.model)
		torch.save(model.state_dict(), "./model/{}.pth".format(epoch), _use_new_zipfile_serialization=True)
		pass


def train_loop(dataloader, model, loss_fn, optimizer, device, model_type):
	model.train()
	for batch, (X, y) in enumerate(dataloader):
		X, y = X.to(device), y.to(device)
		if model_type in REGRESSION:
			y = y.to(torch.float32)
		pred = model(X)
		loss = loss_fn(pred, y)

		#print('batch: ', batch)
		#print('loss: ', loss)
		#Backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()


def eval_loop(dataloader, model, device, model_type):
	model.eval()
	correct = 0
	total = 0
	correct_recycle = 0
	total_recycle = 0
	with torch.no_grad():
		for images, labels in dataloader:
			images, labels = images.to(device), labels.to(device)
			outputs = model(images)
			if model_type in CNN:
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()

				#Check our model performance for predicting recyclable object,
				#default setting is set to true
				if args.check_recycle_acc:
					predicted = predicted.cpu().numpy()
					labels = labels.cpu().numpy()
					for i in range(len(labels)):
						if labels[i] == 1:
							total_recycle += 1
							if predicted[i] == 1:
								correct_recycle += 1

			elif model_type in REGRESSION:
				prob = outputs.cpu().numpy()[0]
				#Set 0.5 as our threshold for choosing between 0 and 1
				pred = 1 if prob > 0.5 else 0
				labels = labels.cpu().numpy()[0]
				if pred == labels:
					correct += 1
				#Check performance for recyclable objects
				if args.check_recycle_acc and labels == 1:
					total_recycle += 1
					if pred == 1:
						correct_recycle += 1

	if args.check_recycle_acc and total_recycle != 0:
		print('Accuracy of recycle: ', correct_recycle / total_recycle)
	print('Accuracy of the network on the %d test images: %d %%' % (len(dataloader.dataset),
		100 * correct / len(dataloader.dataset)))


def parse_args():
	parser = ArgumentParser()
	parser.add_argument(
		"--data_path",
		type=str,
		default="./DATASET/",
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=32,
	)
	parser.add_argument(
		"--num_epoch",
		type=int,
		default=30,
	)
	parser.add_argument(
		"--model",
		type=str,
		default='classifier'
	)
	parser.add_argument(
		"--do_semi",
		type=bool,
		default=False,
	)
	parser.add_argument(
		"--check_recycle_acc",
		type=bool,
		default=True,
	)

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	Path("./model").mkdir(parents=True, exist_ok=True)
	args = parse_args()
	main(args)