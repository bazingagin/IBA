from transformers import *
import numpy as np
import torch
from lime.lime_text import LimeTextExplainer
from captum.attr import IntegratedGradients

from code.evaluations.Degradation import DegradationTest
from code.evaluations.LIME_ import *
from code.evaluations.IntegratedGradient_ import *
from code.utils import *
from code.methods import *

np.random.seed(3)


class EvaluationDegradation:
    def __init__(self, model, tokenizer, method_name, task, special_tokens, test_data, test_num, degrad_num, reverse, device,
                 layer_idx=9, beta=1e-5, lr=1, train_steps=10, wrapper=None, pred_fn=None):
        self.device = device
        self.model = model.to(self.device)  # model to evaluate
        self.tokenizer = tokenizer  # huggingface transformer style tokenizer
        self.method_name = method_name  # method to interpret model
        self.task = task  # task can be classification ('cl') or entailment ('et')
        self.special_tokens = special_tokens
        self.test_data = test_data
        self.test_num = test_num
        self.degrad_num = degrad_num
        self.layer_idx = layer_idx
        self.reverse = reverse
        if self.method_name == 'iba':
            self.saliency_func = layer_heatmap_iba
            self.saliency_func_args = [self.model, self.layer_idx, beta, lr, train_steps]
        elif self.method_name == 'random':
            self.saliency_func = np.random.uniform
            self.saliency_func_args = []
        else:
            raise Exception("Sorry methods not supported.")

    def process(self, max_len=512):
        to_test = np.array(self.test_data)
        # if self.test_num < len(self.test_data):
        to_test_idx = np.random.choice(len(self.test_data), self.test_num, replace=False)
        to_test = to_test[to_test_idx]

        # Start degradation test
        degradation = DegradationTest(self.model, self.degrad_num, self.reverse, self.layer_idx, self.special_tokens)
        if self.task == 'cl':
            for i, test_instance in enumerate(tqdm(to_test)):
                text, target = test_instance[0], int(test_instance[1])
                text_ids = (
                    torch.tensor([self.tokenizer.encode(text, add_special_tokens=True, max_length=max_len)])).to(
                    self.device)
                text_words = self.tokenizer.convert_ids_to_tokens(text_ids[0])

                saliency_args = [text, None, target, text_words, text_ids, None] + self.saliency_func_args
                if self.method_name != 'random':
                    layer_saliency = self.saliency_func(*saliency_args)
                else:
                    layer_saliency = self.saliency_func(size=len(text_words))

                degradation.eval(layer_saliency, text_words, text_ids, target)
        elif self.task == 'et':
            for i, test_instance in enumerate(tqdm(to_test)):
                text1, text2, target = test_instance[0], test_instance[1], int(test_instance[2])
                encoded = self.tokenizer.encode_plus(text=text1, text_pair=text2, return_token_type_ids=True,
                                                     max_length=max_len)
                text_ids = torch.tensor([encoded["input_ids"]]).to(self.device)
                segment_ids = torch.tensor([encoded["token_type_ids"]]).to(self.device)
                text_words = self.tokenizer.convert_ids_to_tokens(text_ids[0])

                saliency_args = [text1, text2, target, text_words, text_ids, segment_ids] + self.saliency_func_args

                if self.method_name != 'random':
                    layer_saliency = self.saliency_func(*saliency_args)
                else:
                    layer_saliency = self.saliency_func(size=len(text_words))

                degradation.eval(layer_saliency, text_words, text_ids, target, seg_ids=segment_ids)
        else:
            raise Exception("Task not supported. Please try 'cl' for classification or 'et' for entailment. ")

        # Calculate result
        original_prob, original_accuracy = degradation.return_original_result()
        degrad_prob, degrad_accuracy = degradation.return_degradation_result()
        original_total_prob = sum(original_prob) / self.test_num
        original_total_accuracy = sum(original_accuracy) / self.test_num
        accumulated_degrad_prob = np.sum(np.array(degrad_prob), axis=0) / self.test_num
        accumulated_degrad_acc = np.sum(np.array(degrad_accuracy), axis=0) / self.test_num
        print("original average accuracy is ", original_total_accuracy)
        print("original average probability is ", original_total_prob)
        print("degradation accuracy is ", accumulated_degrad_acc)
        print("degradation probaility is ", accumulated_degrad_prob)

        return original_total_accuracy, original_prob, accumulated_degrad_acc, accumulated_degrad_prob


def evaluation_degradation(model_state_dict_path,
                           model_pickled_dir,
                           method,
                           task,
                           special_tokens,
                           test_data,
                           test_num,
                           class_num,
                           degrad_num,
                           layer_idx,
                           reverse,
                           device,
                           beta,
                           lr,
                           train_steps):
    # Initialize BERT
    wrapper = None
    pred_fn = None
    if model_pickled_dir is None:
        config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True, num_labels=class_num)
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", config=config)
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model.to(device)
        # Load Weights for BERT
        model.load_state_dict(torch.load(model_state_dict_path, map_location=device))
    else:
        config = BertConfig.from_pretrained(model_pickled_dir, output_hidden_states=True, num_labels=class_num)
        model = BertForSequenceClassification.from_pretrained(model_pickled_dir, config=config)
        tokenizer = BertTokenizer.from_pretrained(model_pickled_dir)
        model.to(device)

    # Initialize Evaluation
    evaluation = EvaluationDegradation(model, tokenizer, method, task, special_tokens, test_data, test_num, degrad_num,
                                       reverse, device,
                                       layer_idx, beta, lr, train_steps, wrapper, pred_fn)
    original_acc, original_prob, degrad_acc, degrad_prob = evaluation.process()
    return original_acc, original_prob, degrad_acc, degrad_prob
