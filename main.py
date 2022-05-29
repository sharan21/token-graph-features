import pickle
import os
import copy
import argparse
from utils import *
from tqdm import tqdm
import numpy as np
import copy
from MulticoreTSNE import MulticoreTSNE as TSNE
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import random

parser = argparse.ArgumentParser()

parser.add_argument('--model-name', default=None, help='name of model dir')
parser.add_argument('--dataset', default=None, help='name of dataset')
parser.add_argument('--update', default=None,
					help='path to file to update graph')
parser.add_argument('--write-mdata', default=False,
					action='store_true', help='display details of checkpoint')
parser.add_argument('--checkpoint', default='default.pkl',
					help='name of checkpoint to update')
parser.add_argument('--bidi', default=True,
					action='store_true', help='bidirectional link updation')
parser.add_argument('--embedding-tsne', default=False,
					action='store_true', help='bidirectional link updation')
parser.add_argument('--create-embds', default=False,
					action='store_true', help='create embedding matrix of dim v*v')
parser.add_argument('--m', type=int, default=100, metavar='N',
					help='num of samples for importance sampling estimate')
parser.add_argument('--window', type=int, default=1, metavar='N',
					help='length of window')

class TokenGraphEmbedding():
	def __init__(self, args):
		self.args = args
		self.num_nodes = 0
		self.tot_weight = 0
		self.i2w = {}
		self.w2i = {}
		self.adj = {}
		self.bidi = args.bidi
		self.window = args.window

		#create/load the model
		self.model_dir = os.path.join('checkpoints', args.dataset, args.model_name)
		if(not os.path.exists(self.model_dir)):  # create new model dir
			os.mkdir(self.model_dir)
			print("Creating model dir {}".format(args.model_name))		
		self.save_path = os.path.join(self.model_dir, args.checkpoint)
		self.load()

	def save(self):
		with open(self.save_path, 'wb') as handle:
			pickle.dump(self, handle)

	def load(self): #load from .pkl file
		try:
			with open(self.save_path, 'rb') as handle:
				b = pickle.load(handle)
				self.init_from_obj(b)
				print("Loaded checkpoint {}".format(self.save_path))
				print(self.args)
		except:
			print("Creating new checkpoint {}".format(self.save_path))

	def create_embds(self):  # create embeddings from self.adj
		assert(self.num_nodes == len(self.adj) == len(self.i2w) == len(self.w2i))
		embds = torch.zeros((self.num_nodes, self.num_nodes)) #create 2d array of 0s of dim v*v
		for word in self.adj:
			i = self.w2i[word]
			for nei in self.adj[word]:
				j = self.w2i[nei]
				embds[i][j] = self.adj[word][nei]
		assert(embds.sum() == self.tot_weight)
		embds = (nn.Softmax(dim=-1))(embds)
		with open(os.path.join(self.model_dir, 'embds.pkl'), 'wb') as handle: #save embds
			pickle.dump(embds.numpy(), handle)
		return embds.numpy()

	def init_from_obj(self, b):  # copy constructors
		self.args = b.args
		self.num_nodes = b.num_nodes
		self.tot_weight = b.tot_weight
		self.i2w = b.i2w
		self.w2i = b.w2i
		self.adj = b.adj
		self.bidi = b.bidi
		self.window = b.window

	def write_mdata(self, write_adj=False):
		with open(os.path.join(t.model_dir, 'mdata.txt'), 'w') as f:
			f.write("num_nodes: {} \n".format(self.num_nodes))
			f.write("tot_weight: {} \n".format(self.tot_weight))
			if(write_adj):
				f.write("\nAdjacency Matrix: \n")
				for token in self.w2i:
					f.write(token + '\n')
					for neigh in self.adj[token]:
						f.write("\t {} : {} \n".format(
							neigh, self.adj[token][neigh]))
					f.write('\n')

	def update_link(self, token_src, token_dst, weight, stop=False):
		if(token_src not in self.adj): # check if this token exists
			self.adj[token_src] = {}
			assert(token_src not in self.w2i)
			self.w2i[token_src] = self.num_nodes
			self.i2w[self.num_nodes] = token_src
			self.num_nodes += 1
		# check if prior link exists
		if token_dst not in self.adj[token_src]:
			self.adj[token_src][token_dst] = 0
		# update dst -> src link
		self.adj[token_src][token_dst] += weight
		self.tot_weight += weight
		if(args.bidi):  # node updates neighbours from left and right
			if(not stop):
				self.update_link(token_dst, token_src, weight, stop=True)

	def update_graph(self, from_file):
		with open(from_file, "r") as f:
			sents = f.readlines()
		for i in tqdm(range(len(sents))):
			sent = sents[i]
			sent = clean_line(sent)
			tokens = sent.split(' ')
			for i in range(len(tokens)):
				for j in range(1, self.window+1):
					if(i-j >= 0):
						self.update_link(token_src=tokens[i], token_dst=tokens[i-j], weight=1)
		self.save()


if __name__ == "__main__":
	args = parser.parse_args()

	# create/load the model
	t = TokenGraphEmbedding(args)
	log_file = open(os.path.join(t.model_dir, 'log.txt'), 'a')

	if(args.update is not None):
		t.update_graph(from_file=args.update)
		t.write_mdata()
		log_file.write("updated from {} \n".format(args.update))
	
	if(args.write_mdata):
		t.write_mdata()
		log_file.write("wrote mdata to dir\n")
		
	if(args.create_embds):
		t.create_embds()
		log_file.write("created embds \n")
	
	if(args.embedding_tsne):
		z = t.create_embds()
		english = [t.i2w[k] for k in t.i2w]
		tsne = TSNE(n_jobs=10)
		res = tsne.fit_transform(z)
		min_x, min_y = np.amin(res, axis=0)
		max_x, max_y = np.amax(res, axis=0)
		print("Done performing TSNE, Plotting...")
		plt.figure(figsize=(15, 15))
		plt.xlim(min_x, max_x)
		plt.ylim(min_y, max_y)
		for i in tqdm(range(args.m)):
			plt.text(res[i,0], res[i,1], english[i], fontsize=random.randint(3, 7), rotation=random.randint(0, 90))
		plt.savefig(os.path.join(t.model_dir, 'tsne-embd.pdf'))
		
	log_file.close()
