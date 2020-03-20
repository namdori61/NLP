import numpy as np
import torch
from torch.utils.data import Dataset

class NPLMDataset(Dataset):
    def __init__(self, corpus_path, context_size):
        self.corpus_path = corpus_path
        self.context_size = context_size
        with open(self.corpus_path,'r') as f:
            self.corpus = f.readlines()

        vocabs = list(set(word for text in self.corpus for word in text.split(' ')))
        self.vocab_size = len(vocabs)
        print("{} unique words in corpus".format(len(vocabs)))
        self.word_to_id = {k: v for v, k in enumerate(vocabs)}
        self.id_to_word = {k: v for k, v in enumerate(vocabs)}
        self.corpus_with_id = []
        for text in self.corpus:
            text_with_id = []
            for word in text.split(' '):
                text_with_id.append(self.word_to_id[word])
            self.corpus_with_id.append(text_with_id)
        print("{} sentences in corpus processed with indexed words".format(len(self.corpus_with_id)))

        dataset = []
        for text in self.corpus_with_id:
            for i in range(self.context_size, len(text)):
                first_context_idx = i - self.context_size
                instance = []
                for j in range(first_context_idx, i):
                    instance.append(text[j])
                instance.append(text[i])
                dataset.append(instance)
        self.data = torch.LongTensor(np.array(dataset))
        print("{} instances in dataset".format(len(self.data)))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        target = self.data[index][-1]
        contexts = self.data[index][:-1]
        return contexts, target