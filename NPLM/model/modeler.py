import numpy as np
import torch
import torch.nn as nn

class NPLM(nn.Module):
    """
    embedding: Embedding for context word.
    """
    def __init__(self, embedding_size, vocab_size, context_size, hidden_size, batch_size):
        super(NPLM, self).__init__()
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        
        self.embeddings = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embeddings.weight.data.uniform_(-1, 1)
        self.H = nn.Linear(self.embedding_size * self.context_size, self.hidden_size, bias=False)
        self.H.weight.data.uniform_(-1, 1)
        self.U = nn.Linear(self.hidden_size, self.vocab_size, bias=False)
        self.U.weight.data.uniform_(-1, 1)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

    def forward(self, context_words):
        context_embedding = self.embeddings(context_words)
        x = context_embedding.view(self.batch_size, -1)
        hidden_in = self.H(x)
        hidden_out = self.tanh(hidden_in)
        score = self.U(hidden_out)
        probs = self.softmax(score)
        return probs