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

parser = argparse.ArgumentParser()
parser.add_argument('--create', default=None, help='name of model to create')
parser.add_argument('--load', default=None, help='name of model to load')
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

class TokenGraphEmbedding():
	def __init__(self, args):
		self.args = args
		self.num_nodes = 0
		self.tot_weight = 0
		self.i2w = {}
		self.adj = {}
		self.w2i = {}
		self.bidi = args.bidi

		if(args.create is not None):
			print("Creating model {}".format(args.create))
			self.model_dir = os.path.join('checkpoints', args.create)
			if(not os.path.exists(self.model_dir)):  # create new model dir
				os.mkdir(self.model_dir)
			else:
				exit("Model dir already exists, load instead.")
			self.save_path = os.path.join(self.model_dir, args.checkpoint)
		else:
			self.model_dir = os.path.join('checkpoints', args.load)
			self.save_path = os.path.join(self.model_dir, args.checkpoint)
			print("Loading model from {}".format(self.save_path))
			self.load()

	def save(self):
		with open(self.save_path, 'wb') as handle:
			pickle.dump(self, handle)

	def load(self):
		try:
			with open(self.save_path, 'rb') as handle:
				b = pickle.load(handle)
		except:
			print("failed to load checkpoint, updating with new object.")
			return
		self.init_from_obj(b)

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
		return embds.numpy()

	def init_from_obj(self, b):  # copy constructors
		self.num_nodes = b.num_nodes
		self.i2w = b.i2w
		self.adj = b.adj
		self.tot_weight = b.tot_weight
		self.w2i = b.w2i
		self.bidi = b.bidi

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
		for sent in sents:
			sent = clean_line(sent)
			tokens = sent.split(' ')
			for i in tqdm(range(len(tokens))):
				if(i == 0):
					continue
				self.update_link(
					token_src=tokens[i], token_dst=tokens[i-1], weight=1)
		self.save()


if __name__ == "__main__":
	args = parser.parse_args()
	assert(not(args.create and args.load))
	assert(args.create or args.load)

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
		print("Done performing TSNE, Plotting...")
		plt.figure(figsize=(15, 15))
		plt.xlim(-10, 10)
		plt.ylim(-10, 10)

		for i in tqdm(range(args.m)):
			plt.text(res[i,0], res[i,1], english[i], fontsize=7, rotation=30)
		plt.savefig(os.path.join(t.model_dir, 'tsne-embd.png'))
		
	log_file.close()
