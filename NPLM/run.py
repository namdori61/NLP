import sys, os, argparse
import torch
import torch.nn as nn
from torch.optim  import Adam
from model.data_loader import NPLMDataset
from torch.utils.data import DataLoader
from model.modeler import NPLM
from trainer import trainer
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Config file path')
    parser.add_argument('--corpus_path', type=str, help='Data')
    args = parser.parse_args()

    config, _ = get_config_from_json(args.config_path)

    dataset = NPLMDataset(args.corpus_path, config.context_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    kwargs = {'num_workers': 1, 'pin_memory': True} if device=='cuda' else {}
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, **kwargs)

    model = NPLM(config.embedding_size, dataset.vocab_size, config.context_size, config.hidden_size, config.batch_size)
    model = model.to(device)

    loss_func = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    trainer(model, loss_func, optimizer, dataloader, device, config.num_epoch)