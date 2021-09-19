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
import sys


def main(args):
	train_tfm = transforms.Compose([
		transforms.Resize((128, 128)),
		# You may add some transforms here.
		#transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
		transforms.RandomRotation(degrees=(0, 180)),
		transforms.Pad(padding=10),
		transforms.ToTensor(),
	])

	test_tfm = transforms.Compose([
		transforms.Resize((128, 128)),
		transforms.ToTensor(),
	])

	train_set = ImageFolder(args.data_path + 'TRAIN', transform=train_tfm)
	trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
	eval_set = ImageFolder(args.data_path + 'TEST', transform=test_tfm)
	evalloader = DataLoader(eval_set, batch_size=args.batch_size, shuffle=False)
	#train_imshow(trainloader)
	#test_set
	#for x in train_set:
	#	print(x)
	#	sys.exit()

	device = "cuda" if torch.cuda.is_available() else "cpu"
	torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
	model = Classifier().to(device) if args.model == 'Classifier' else torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
	loss_fn = nn.CrossEntropyLoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01)

	epoch_pbar = trange(args.num_epoch, desc="Epoch")
	for epoch in epoch_pbar:
		train_loop(trainloader, model, loss_fn, optimizer, device)
		if args.do_semi:
			# Obtain pseudo-labels for unlabeled data using trained model.
			pseudo_set = get_pseudo_labels(eval_set, model, device, threshold=0.8)

			# Construct a new dataset and a data loader for training.
			# This is used in semi-supervised learning only.
			concat_dataset = ConcatDataset([train_set, pseudo_set])
			train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True)
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

def get_pseudo_labels(dataset, model, device, threshold=0.65):
	# This functions generates pseudo-labels of a dataset using given model.
	# It returns an instance of DatasetFolder containing images whose prediction confidences exceed a given threshold.
	device = "cuda" if torch.cuda.is_available() else "cpu"
	# Construct a data loader.
	data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
	# Make sure the model is in eval mode.
	model.eval()
	# Define softmax function.
	softmax = nn.Softmax(dim=-1)
	# Iterate over the dataset by batches.
	for batch in tqdm(data_loader):
		img, _ = batch

		# Forward the data
		# Using torch.no_grad() accelerates the forward process.
		with torch.no_grad():
			logits = model(img.to(device))

		# Obtain the probability distributions by applying softmax on logits.
		probs = softmax(logits)
		print('probs: ', probs)
		for prob in probs:
			print('prob: ', prob)
			print('argmax: ', prob.argmax(dim=0))

		# Filter the data and construct a new dataset.
		newdata = tuple(compress(img, [prob[torch.argmax(prob)] > threshold for prob in probs]),
						 prob.argmax(dim=0))
		print(newdata[0])
		sys.exit()
		#dataset.extend(newdata)

	# # Turn off the eval mode.
	model.train()
	return pseudo_set


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
		default=128,
	)
	parser.add_argument(
		"--num_epoch",
		type=int,
		default=10,
	)
	parser.add_argument(
		"--model",
		type=str,
		default='Classifier'
	)
	parser.add_argument(
		"--do_semi",
		type=bool,
		default=False,
	)

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = parse_args()
	main(args)