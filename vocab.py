from collections import Counter

class Vocab(object):
    def __init__(self, path):
        self.word2idx = {}
        self.idx2word = []

        with open(path) as f:
            i = 0
            for line in f:
                w = line.split()[0]
                self.word2idx[w] = len(self.word2idx)
                self.idx2word.append(w)
                i += 1

        self.size = len(self.word2idx)
        self.pad = self.word2idx['<pad>']
        self.go = self.word2idx['<go>']
        self.eos = self.word2idx['<eos>']
        self.unk = self.word2idx['<unk>']
        self.blank = self.word2idx['<blank>']
    
    @staticmethod
    def clean_line(line, bad=[',','.', ';', '(', ')', '/', '`', '%', '"', '-', '\\','\'',]): # use this function carefully
        clean = ''
        for c in line:
            if c not in bad:
                clean += c
        return clean
    
    @staticmethod
    def build(sents, path, size):
        v = ['<pad>', '<go>', '<eos>', '<unk>', '<blank>']
        words = [w for s in sents for w in s]
        cnt = Counter(words)
        print('# distinct words found: {}'.format(len(cnt)))
        n_unk = len(words)
        for w, c in cnt.most_common(size):
            if(w in v):
                continue
            v.append(w)
            n_unk -= c
        cnt['<unk>'] = n_unk

        with open(path, 'w') as f:
            for w in v:
                f.write('{}\t{}\n'.format(w, cnt[w]))
    
    def get_sent(self, sents, clean=False):
        eng = []
        for i in range(len(sents)):
            sent = [self.idx2word[word] for word in sents[i]]
            line = clean_line(' '.join(sent)) if clean else ' '.join(sent)
            eng.append(line)
        return eng

