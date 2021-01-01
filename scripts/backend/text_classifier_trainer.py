import torch
from torchtext import data
from data_prepare import load_text_vocab
from tqdm import tqdm


def binary_accuracy(prediction, label):
    rounded_prediction = torch.round(torch.sigmoid(prediction))
    correct = (rounded_prediction == label).float()
    accuracy = correct.sum()/len(correct)
    return accuracy


class text_classifier_trainer:
    def __init__(self, model, criterion, optimizer, text_path, batch_size, train_data, valid_data, test_data, device):
        self.device = device
        self.model = model.to(device)
        self.text = load_text_vocab(text_path)
        self.pretrained_embeddings = self.text.vocab.vectors
        self.model.embedding.weight.data.copy_(self.pretrained_embeddings)
        self.unk_idx = self.text.vocab.stoi[self.text.unk_token]
        self.model.embedding.weight.data[self.unk_idx] = torch.zeros(self.model.embedding_dim)
        self.model.embedding.weight.data[self.model.pad_idx] = torch.zeros(self.model.embedding_dim)
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.train_iterator, self.valid_iterator, self.test_iterator = self.construct_batches()
    

    def save_model(self, path):
        torch.save(self.model, path)
    

    def save_weights(self, path):
        torch.save(self.model.state_dict(), path)
    

    def load_weights(self, path):
        self.model.load_state_dict(torch.load(path))
    

    def save_history(self, train_path, valid_path):
        torch.save((self.train_batch_history, self.train_epoch_history), train_path)
        torch.save((self.valid_batch_history, self.valid_epoch_history), valid_path)
    

    def load_history(self, train_path, valid_path):
        self.train_batch_history, self.train_epoch_history = torch.load(train_path)
        self.valid_batch_history, self.valid_epoch_history = torch.load(valid_path)


    def get_history(self):
        return self.train_batch_history, self.valid_batch_history, self.train_epoch_history, self.valid_epoch_history


    def construct_batches(self):
        train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits((self.train_data, self.valid_data, self.test_data),
                                                                                   batch_size=self.batch_size,
                                                                                   device=self.device)
        return train_iterator, valid_iterator, test_iterator


    def iterate_batches(self, iterator, train, mode):
        batch_loss = []
        batch_accuracy = []
        i = 0
        batch_range = tqdm(iterator, desc='')
        for batch in batch_range:
            if train:
                self.optimizer.zero_grad()
            prediction = self.model(batch.text).squeeze(1)
            loss = self.criterion(prediction, batch.label)
            accuracy = binary_accuracy(prediction, batch.label)
            if train:
                loss.backward()
                self.optimizer.step()
            batch_loss.append(loss.item())
            batch_accuracy.append(accuracy.item())
            i += 1
            epoch_loss = sum(batch_loss)/i
            epoch_accuracy = sum(batch_accuracy)/i
            batch_range.set_description('| {} | loss: {:.4f} | accuracy: {:.4f} |'.format(mode, epoch_loss, epoch_accuracy))
        return batch_loss, batch_accuracy, epoch_loss, epoch_accuracy       


    def train_evaluate_epoch(self, iterator, train, mode):
        if train:
            self.model.train()
            batch_loss, batch_accuracy, epoch_loss, epoch_accuracy = self.iterate_batches(iterator, train, mode)
        else:
            self.model.eval()
            with torch.no_grad():
                batch_loss, batch_accuracy, epoch_loss, epoch_accuracy = self.iterate_batches(iterator, train, mode)
        return batch_loss, batch_accuracy, epoch_loss, epoch_accuracy
    

    def train(self, n_epoch, weight_path, train_path, valid_path):
        self.train_batch_history = []
        self.valid_batch_history = []
        self.train_epoch_history = []
        self.valid_epoch_history = []
        lowest_valid_epoch_loss = float('inf')
        for epoch in range(n_epoch):
            train_batch_loss, train_batch_accuracy, train_epoch_loss, train_epoch_accuracy = self.train_evaluate_epoch(self.train_iterator, True, 'train')
            valid_batch_loss, valid_batch_accuracy, valid_epoch_loss, valid_epoch_accuracy = self.train_evaluate_epoch(self.valid_iterator, False, 'validate')
            self.train_batch_history.append([train_batch_loss, train_batch_accuracy])
            self.valid_batch_history.append([valid_batch_loss, valid_batch_accuracy])
            self.train_epoch_history.append([train_epoch_loss, train_epoch_accuracy])
            self.valid_epoch_history.append([valid_epoch_loss, valid_epoch_accuracy])
            if valid_epoch_loss < lowest_valid_epoch_loss:
                lowest_valid_epoch_loss = valid_epoch_loss
                self.save_weights(weight_path)
        self.save_history(train_path, valid_path)
    

    def test(self):
        return self.train_evaluate_epoch(self.test_iterator, False, 'test')
