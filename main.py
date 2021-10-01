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
from classifier import Classifier
from itertools import compress
from pathlib import Path
import sys


def main(args):
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

	train_tfm = transforms.Compose([
		transforms.Resize((224, 224)),
		# You may add some transforms here.
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

	#args.data_path = ./DATASET/
	train_set = ImageFolder(args.data_path + 'TRAIN', transform=train_tfm)
	eval_set = ImageFolder(args.data_path + 'TEST', transform=test_tfm)


	train_set = [(X, torch.tensor(y)) for (X, y) in train_set]
	eval_set = [(X, torch.tensor(y)) for (X, y) in eval_set]
	trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
	evalloader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False)

	device = "cuda" if torch.cuda.is_available() else "cpu"
	#print(device)
	#create model
	torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
	model = Classifier().to(device)
	if args.model == 'resnet18':
		model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True) 
    	#for param in model.parameters():
    	#	param.requires_grad = False
		model.fc = nn.Sequential(
					nn.Linear(512, 128),
					nn.ReLU(),
					nn.Linear(128, 2)).to(device)


	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

	epoch_pbar = trange(args.num_epoch, desc="Epoch")
	for epoch in epoch_pbar:
		train_loop(trainloader, model, loss_fn, optimizer, device)
		if args.do_semi:
			# Obtain pseudo-labels for unlabeled data using trained model.
			pseudo_set = get_pseudo_labels(eval_set, model, device, threshold=0.8)
			# Construct a new dataset and a data loader for training.
			# This is used in semi-supervised learning only.
			concat_dataset = ConcatDataset([train_set, pseudo_set])
			trainloader = DataLoader(concat_dataset, batch_size=args.batch_size, shuffle=True)
		eval_loop(evalloader, model, device)
		torch.save(model.state_dict(), "./model/{}".format(epoch))
		pass


def train_loop(dataloader, model, loss_fn, optimizer, device):
	model.train()
	for batch, (X, y) in enumerate(dataloader):
		X, y = X.to(device), y.to(device)
		pred = model(X)
		loss = loss_fn(pred, y)
		print('batch: ', batch)
		print('loss: ', loss)
		#Backpropagation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

def eval_loop(dataloader, model, device):
	model.eval()
	correct = 0
	total = 0
	with torch.no_grad():
		for images, labels in dataloader:
			outputs = model(images)
			_, predicted = torch.max(outputs.data, 1)
			total += labels.size(0)
			correct += (predicted == labels).sum().item()
			#print('total: ', total)
			#print('correct: ', correct)
	print('Accuracy of the network on the %d test images: %d %%' % (len(dataloader.dataset),
		100 * correct / total))

def train_imshow(loader):
	classes = ('O', 'R')
	dataiter = iter(loader)
	images, labels = dataiter.next()
	fig, axes = plt.subplots(figsize=(10, 4), ncols=5)
	for i in range(5):
		ax = axes[i]
		ax.imshow(images[i].permute(1, 2, 0))  # permute?
		ax.title.set_text(' '.join('%5s' % classes[labels[i]]))
		print(images[i].shape)  # Not needed
	plt.show()
	print('images shape on batch size = {}'.format(images.size()))
	print('labels shape on batch size = {}'.format(labels.size()))

def get_pseudo_labels(dataset, model, device, threshold=0.8):
	# This functions generates pseudo-labels of a dataset using given model.
	# It returns an instance of DatasetFolder containing images whose prediction confidences exceed a given threshold.
	# Construct a data loader.
	data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
	# Make sure the model is in eval mode.
	model.eval()
	# Define softmax function.
	softmax = nn.Softmax(dim=-1)
	# Iterate over the dataset by batches.
	for batch in tqdm(data_loader):
		img, _ = batch

		with torch.no_grad():
			logits = model(img.to(device))

		# Obtain the probability distributions by applying softmax on logits.
		probs = softmax(logits)

		# Filter the data and construct a new dataset.
		pseudo_set = []
		for i in range(len(img)):
			if probs[i].argmax(dim=0) > threshold:
				pseudo_set.append(tuple([img[i], probs[i].argmax(dim=0)] ) )

	# # Turn off the eval mode.
	model.train()
	return tuple(pseudo_set)


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

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	Path("./model").mkdir(parents=True, exist_ok=True)
	args = parse_args()
	main(args)