from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from data_prepare import preprocess_data, load_text_vocab, load_label_vocab
from text_classifier_model import text_classifier_model
from text_classifier_trainer import text_classifier_trainer
from text_classifier_predictor import text_classifier_predictor

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

new_calculation = False
seed = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

data_path = Path(__file__).parent / '../../.data'
vector_path = Path(__file__).parent / '../../.vector_cache'
text_path = Path(__file__).parent / '../../model/vocab/text_vocab.pt'
label_path = Path(__file__).parent / '../../model/vocab/label_vocab.pt'
model_path = Path(__file__).parent / '../../model/model.pt'
weight_path = Path(__file__).parent / '../../model/weight/weight.pt'
train_path = Path(__file__).parent / '../../model/history/train.pt'
valid_path = Path(__file__).parent / '../../model/history/valid.pt'

print('preprocessing data')
train_data, valid_data, test_data = preprocess_data(data_path, vector_path, text_path, label_path, seed)

text = load_text_vocab(text_path)
label = load_text_vocab(label_path)
text_vocab_size = len(text.vocab)
label_vocab_size = len(label.vocab)
print('unique tokens in text vocabulary: {}'.format(text_vocab_size))
print('unique tokens in label vocabulary: {}'.format(label_vocab_size))
print('10 most frequent words in text vocabulary: '+(10*'{} ').format(*text.vocab.freqs.most_common(10)))
print('labels: {}'.format(str(label.vocab.stoi)))

input_dim = text_vocab_size
embedding_dim = 100
output_dim = 1
pad_idx = text.vocab.stoi[text.pad_token]

model = text_classifier_model(input_dim, embedding_dim, output_dim, pad_idx)
n_param = count_parameters(model)
print('text_classifier_model initialized with {} trainable parameters'.format(n_param))

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())
batch_size = 64
n_epoch = 5
trainer = text_classifier_trainer(model, criterion, optimizer, text_path, batch_size, train_data, valid_data, test_data, device)

if new_calculation:
    print('training text_classifier_model')
    trainer.train(n_epoch, weight_path, train_path, valid_path)
    trainer.save_model(model_path)
else:
    trainer.load_weights(weight_path)
    trainer.load_history(train_path, valid_path)

test_batch_loss, test_batch_accuracy, test_epoch_loss, test_epoch_accuracy = trainer.test()
print('\ntest loss: {}\ntest accuracy: {}'.format(test_epoch_loss, test_epoch_accuracy))

predictor = text_classifier_predictor(device, text_path, label_path, model_path, weight_path)
sample_bad, sample_good = 'this film is terrible', 'this film is great'
out_bad, out_good = predictor.predict(sample_bad), predictor.predict(sample_good)
print('\n{}\n{}\n\n{}\n{}'.format(sample_bad, out_bad, sample_good, out_good))
