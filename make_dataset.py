import numpy as np

class Corpus:
    def __init__(self, filenames, punc = """`!&*()?><.,;[]{}"""):
        self.punc = punc
        self.corpus = [self.tokenize(open(f).read()) for f in filenames]

        self.chrs = set("""0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!@#$%^&*()-_=+|`:;'"/?.,><[]{}""")
        self.chrs.add("<w>")  # word start
        self.chrs.add("</w>") # word stop
        self.chr2idx = {c:i for i,c in enumerate(self.chrs)}
        self.idx2chr = {i:c for i,c in enumerate(self.chrs)}

        self.words = self._get_words()
        self.word2idx = {c:i for i,c in enumerate(self.words)}
        self.idx2word = {i:c for i,c in enumerate(self.words)}

        self.max_wordlen = self._max_wordlen()

        self.masking_value = -1
        self.T = self.max_wordlen + 2 # add two for start and stop
        self.D = len(self.chrs)
        self.W = len(self.words)

    def _max_wordlen(self):
        mx = 0
        for doc in self.corpus:
            mx = reduce(lambda cur_mx, val: max(cur_mx, len(val)), doc.split(), mx)
        return mx

    def _get_words(self):
        words = set()
        for doc in self.corpus:
            for word in doc.split():
                words.add(word)
        return words

    def idx2chr(self, i):
        return self.idx2chr[i]

    def idx2chr(self, c):
        return self.chr2idx[c]

    def tokenize(self, text):
        new_text = text
        for p in self.punc:
            new_text = new_text.replace(p," " + p + " ")
        return " ".join(new_text.split())

    def __len__(self):
        return len(self.corpus)

    def getWords(self, doc_num):
        return self.corpus[doc_num].split()

    def docChars2OneHot(self, doc_num):
        words = self.getWords(doc_num)
        N = len(words)
        T = self.T
        D = self.D
        mat = np.zeros((N, T, D))
        mat[:, 0, self.chr2idx["<w>"]] = 1;
        for i, word in enumerate(words):
            for j, char in enumerate(word):
                mat[i, j+1, self.chr2idx[char]] = 1
            mat[i, j+2, self.chr2idx["</w>"]] = 1
            if j+3 < T:
                mat[i, j+3:, :] = self.masking_value # mask out the rest
        return mat

    def docWords2OneHot(self, doc_num):
        doc_words = self.getWords(doc_num)
        N = len(doc_words)
        D = len(self.words)
        mat = np.zeros((N, D))
        for i, word in enumerate(doc_words):
            mat[i, self.word2idx[word]] = 1
        return mat
