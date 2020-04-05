from abc import *
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import collections

class AbstractDataSet(Dataset, metaclass=ABCMeta):
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

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, index):
        pass


class CBOWDataset(AbstractDataSet):
    def __init__(self, corpus_path, context_size):
        super().__init__(corpus_path, context_size)
        dataset = []
        for text in self.corpus_with_id:
            for i in range(self.context_size, len(text) - self.context_size):
                first_context_idx = i - self.context_size
                last_context_idx = i + self.context_size
                instance = []
                for j in range(first_context_idx, last_context_idx + 1):
                    if j != i:
                        instance.append(text[j])
                instance.append(text[i])
                dataset.append(instance)

        self.data = torch.LongTensor(np.array(dataset))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        output = self.data[index][-1]
        inputs = self.data[index][:-1]
        return inputs, output

class SkipGramDataset(AbstractDataSet):
    def __init__(self, corpus_path, context_size):
        super().__init__(corpus_path, context_size)
        dataset = []
        for text in self.corpus_with_id:
            for i in range(self.context_size, len(text) - self.context_size):
                first_context_idx = i - self.context_size
                last_context_idx = i + self.context_size
                for j in range(first_context_idx, last_context_idx + 1):
                    instance = []
                    if j != i:
                        instance.append(text[i])
                        instance.append(text[j])
                        dataset.append(instance)
        
        self.data = torch.LongTensor(np.array(dataset))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        output = self.data[index][1]
        input = self.data[index][0]
        return input, output

class NegSampleDataset(AbstractDataSet):
    def __init__(self, corpus_path, context_size, negative_sample_size=5):
        super().__init__(corpus_path, context_size)
        self.negative_sample_size = negative_sample_size

        total_vocabs = list(word for text in self.corpus for word in text.split(' '))
        word_counter = collections.Counter(total_vocabs)
        word_counter = {self.word_to_id[k]: v / len(total_vocabs) for k, v in word_counter.items()}
        prob_total = sum([v**(3/4) for v in word_counter.values()])
        self.word_probs = {k: v**(3/4) / prob_total for k, v in word_counter.items()}

        dataset = []
        for text in self.corpus_with_id:
            for i in range(self.context_size, len(text) - self.context_size):
                first_context_idx = i - self.context_size
                last_context_idx = i + self.context_size
                
                #positive sampling
                ps = random.choice([j for j in range(first_context_idx, last_context_idx + 1) if j != i])
                dataset.append([i, ps, 1])

                #negative sampling
                neg_samples = np.random.choice(list(self.word_probs.keys()), self.negative_sample_size, p=list(self.word_probs.values()))
                for ns in neg_samples:
                    dataset.append([i, ns, 0])

        self.data = torch.LongTensor(np.array(dataset))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        output = self.data[index][-1]
        inputs = self.data[index][:-1]
        return inputs, output