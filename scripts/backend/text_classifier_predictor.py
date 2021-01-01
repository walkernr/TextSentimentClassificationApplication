import torch
import spacy
from data_prepare import generate_bigrams, load_text_vocab, load_label_vocab

class text_classifier_predictor:
    def __init__(self, device, text_path, label_path, model_path, weight_path):
        self.model = torch.load(model_path)
        self.model.load_state_dict(torch.load(weight_path))
        self.model = self.model.to(device)
        self.model.eval()
        self.text = load_text_vocab(text_path)
        self.label = load_label_vocab(label_path)
        label_dict = self.label.vocab.stoi
        label_dict['positive'] = label_dict.pop('pos')
        label_dict['negative'] = label_dict.pop('neg')
        self.sentiment = dict([(value, key) for key, value in self.label.vocab.stoi.items()])
        self.device = device
        self.nlp = spacy.load('en')


    def predict(self, sample):
        tokenized = generate_bigrams([tok.text for tok in self.nlp.tokenizer(sample)])
        indexed = [self.text.vocab.stoi[t] for t in tokenized]
        tensor = torch.LongTensor(indexed).to(self.device)
        tensor = tensor.unsqueeze(1)
        prediction = torch.sigmoid(self.model(tensor))
        rounded_prediction = torch.round(prediction)
        if prediction.item() < 0.5:
            probability = 1-prediction.item()
        else:
            probability = prediction.item()
        sentiment = self.sentiment[int(rounded_prediction.item())]
        return '{} with {:.2f}% probability'.format(sentiment, 100*probability)
