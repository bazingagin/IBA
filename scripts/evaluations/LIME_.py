import numpy as np
import torch
from transformers import *
from scipy.special import softmax
from operator import methodcaller

class LIME_BERT:
    def __init__(self, model_state_dict_path, model_pickled_dir, class_num):
        # Initialize BERT
        if model_pickled_dir is None:
            config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True, num_labels=class_num)
            self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased", config=config)
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            # Load Weights for BERT
            self.model.load_state_dict(torch.load(model_state_dict_path, map_location=self.device))
            print("Model Loaded")
            self.model.to(self.device)
        else:
            config = BertConfig.from_pretrained(model_pickled_dir, output_hidden_states=True, num_labels=class_num)
            self.model = BertForSequenceClassification.from_pretrained(model_pickled_dir, config=config)
            self.tokenizer = BertTokenizer.from_pretrained(model_pickled_dir)
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
    def predict_prob(self, text):
        with torch.no_grad():
            if ' ****** ' in text:
                text = map(tuple, map(methodcaller("split", " ****** "), text))
                encoded = self.tokenizer.batch_encode_plus(list(text), return_tensors='pt', max_length=512, pad_to_max_length=True)
                text_ids = encoded["input_ids"].to(self.device)
                segment_ids = encoded["token_type_ids"].to(self.device)
                output = self.model(text_ids, token_type_ids=segment_ids)[0]
                return softmax(output.cpu().detach().numpy(), axis=1)
            else:
                encoded = self.tokenizer.batch_encode_plus(text, return_tensors='pt', max_length=512, pad_to_max_length=True)["input_ids"].to(self.device)
                output = self.model(encoded)[0]
                return softmax(output.cpu().detach().numpy(), axis=1)


def generate_lime_attribution(exp, text_words):
    word_attribution = np.array(exp.as_list())[:, 1]
    maximum = max(np.array(word_attribution, dtype='f'))
    word_score = {}
    for w,s in exp.as_list():
        word_score[w.lower()] = s
    rearrange_word_attribution = []
    i = 0
    while i < len(text_words):
        t = text_words[i]
        if t in word_score:
            rearrange_word_attribution.append(word_score[t])
            i+=1
        else:
            if i!=len(text_words)-1 and text_words[i+1].startswith("##") and (t+text_words[i+1].replace("##", "") in word_score):
                combined = t+text_words[i+1].replace("##", "")
                rearrange_word_attribution.append(word_score[combined])
                rearrange_word_attribution.append(word_score[combined])
                i+=2
            else:
                rearrange_word_attribution.append(maximum)
                i+=1
    assert(len(rearrange_word_attribution)==len(text_words))
    return np.array(rearrange_word_attribution)