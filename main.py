import pickle
import os
import copy
import argparse
from utils import *
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--create', default=None, help='name of model to create')
parser.add_argument('--load', default=None, help='name of model to load')
parser.add_argument('--update', default=None, help='path to file to update graph')
parser.add_argument('--display', default=False, action='store_true', help='display details of checkpoint')
parser.add_argument('--checkpoint', default='default.pkl', help='name of checkpoint to update')

class TokenGraphEmbedding():
    def __init__(self, args):    
        self.num_nodes = 0
        self.tot_weight = 0
        self.all_tokens = []  
        self.adj = {} 
        self.w2i = {}

        if(args.create is not None):
            print("Creating model")
            self.model_dir = os.path.join('checkpoints', args.create)
            if(not os.path.exists(self.model_dir)): #create new model dir
                os.mkdir(self.model_dir)
            else:
                exit("Model dir already exists, load instead.")
            self.save_path = os.path.join(self.model_dir, args.checkpoint)
            self.log_file = open(os.path.join(self.model_dir, 'log.txt'), 'a')
        else: 
            print("Loading model")
            self.model_dir = os.path.join('checkpoints', args.load)
            self.save_path = os.path.join(self.model_dir, args.checkpoint)
            self.log_file = open(os.path.join(self.model_dir, 'log.txt'), 'a')
            self.load()

    def save(self):
        with open(self.save_path, 'wb') as handle:
            print(type(self))
            pickle.dump(self, handle)
    
    def load(self):
        with open(self.save_path, 'rb') as handle:
            try:
                b = pickle.load(handle)
            except:
                exit("failed to load checkpoint, ensure the checkpoint file is not empty.")
        self.init_from_obj(b)

    #copy constructor    
    def init_from_obj(self, b):
        self.num_nodes = b.num_nodes
        self.all_tokens = b.all_tokens
        self.adj = b.adj
        self.tot_weight = b.tot_weight
    
    def display(self, print_adj=False):
        print("num_nodes: {}".format(self.num_nodes))
        print("tot_weight: {}".format(self.tot_weight))
        print("all_tokens: {}".format(self.all_tokens))
        
        if(print_adj):
            print("\nAdjacency Matrix: \n")
            for token in self.all_tokens:
                print(token)
                for neigh in self.adj[token]:
                    print("\t {} : {}".format(neigh,self.adj[token][neigh]))
                print()

    def update_link(self, token_src, token_dst, weight, link_type="bidirectional",stop=False):
        #check if this token exists
        if(token_src not in self.adj):
            self.adj[token_src] = {}
            self.all_tokens.append(token_src)
            self.w2i[token_src] = self.num_nodes
            self.num_nodes += 1
        #check if prior link exists
        if token_dst not in self.adj[token_src]:
            self.adj[token_src][token_dst] = 0
        # update dst -> src link
        self.adj[token_src][token_dst] += weight     
        self.tot_weight += weight   
        if(link_type=="bidirectional"):
            if(not stop):
                self.update_link(token_dst, token_src, weight, stop=True)

    def update_graph(self, from_file):
        with open(from_file, "r") as f:
            sents = f.readlines()

        for sent in sents:
            sent = self.clean_line(sent)
            tokens = sent.split(' ')
            for i in tqdm(range(len(tokens))):
                if(i == 0):
                    continue
                self.update_link(token_src=tokens[i], token_dst=tokens[i-1], weight=1)
        t.save()
    
if __name__ == "__main__":
    args = parser.parse_args()
    assert(not(args.create and args.load))
    assert(args.create or args.load)

    #create/load the model
    t = TokenGraphEmbedding(args)
    
    if(args.update is not None):
        t.update_graph(from_file=args.update)

    if(args.display):
        t.display()
    

    
