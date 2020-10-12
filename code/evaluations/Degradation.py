import numpy as np
import torch
from scipy.special import softmax

class DegradationTest:
    def __init__(self, model, degrade_step, reverse, layer_idx, special_tokens, batchify=False):
        self.model = model
        self.step = degrade_step
        self.reverse = reverse
        self.layer_idx = layer_idx
        self.special_tokens = special_tokens
        self.batchify = batchify
        self.original_prob = [] # 1d
        self.original_acc = [] # 1d
        self.degradation_prob = [] # 2d, every element[][] represents probability for truth class at each step
        self.degradation_acc = [] # 2d, every element[][] represents "true"(1) or "false"(0) at each step 

    def _truncate_words(self, sorted_idx, text_words, text_ids, replaced_num, seg_ids=None):
        to_be_replaced_idx = []
        i= 0
        while len(to_be_replaced_idx) < replaced_num and i!=len(text_words)-1:
            current_idx = sorted_idx[i]
            if text_words[current_idx] not in self.special_tokens:
                to_be_replaced_idx.append(current_idx)
            i += 1
        remaining_idx = sorted(list(set(sorted_idx) - set(to_be_replaced_idx)))
        truncated_text_ids = text_ids[0, np.array(remaining_idx)]
        if seg_ids is not None:
            seg_ids = seg_ids[0, np.array(remaining_idx)]
        truncated_text_words = np.array(text_words)[remaining_idx]
        return truncated_text_ids.unsqueeze(0), truncated_text_words, seg_ids

    def _replace_words(self, sorted_idx, text_words, text_ids, replaced_num, mask, mask_id):
        to_be_replaced_idx = []
        i= 0
        while len(to_be_replaced_idx) < replaced_num and i!=len(text_words)-1:
            current_idx = sorted_idx[i]
            if text_words[current_idx] not in self.special_tokens:
                to_be_replaced_idx.append(current_idx)
            i += 1
        replaced_text_ids = text_ids.clone()
        replaced_text_ids[0, to_be_replaced_idx] = mask_id
        replaced_text_words = np.copy(text_words)
        replaced_text_words[to_be_replaced_idx] = mask
        return replaced_text_ids, replaced_text_words

    def predict(self, text_ids, target, att_mask=None, seg_ids=None):
        out = self.model(text_ids, attention_mask=att_mask, token_type_ids=seg_ids)
        prob = out[0]
        pred_class = torch.argmax(prob, axis=1).cpu().detach().numpy()
        pred_class_prob = softmax(prob.cpu().detach().numpy(), axis=1)
        return pred_class[0], pred_class_prob[:, target][0]

    def eval(self, attribution, text_words, text_ids, target, seg_ids=None):
        trunc_ids = []
        total_len = len(text_words)
        granularity = np.linspace(0, 1, self.step)
        trunc_words_num = [int(g) for g in np.round(granularity*total_len)]
        trunc_words_num = list(dict.fromkeys(trunc_words_num))
        if len(trunc_words_num) < self.step:
            # pad till the step length
            trunc_words_num += [trunc_words_num[-1]]*(self.step - len(trunc_words_num))

        # sort idx from high importance to low importance by default
        sorted_idx = np.argsort(-attribution)
        if self.reverse:
            sorted_idx = np.argsort(attribution)

        with torch.no_grad():
            original_class, original_prob = self.predict(text_ids, target, seg_ids=seg_ids)
            self.original_prob.append(original_prob)
            self.original_acc.append(original_class==target)

            instance_degradation_prob = []
            instance_degradation_acc = []
            for num in trunc_words_num[1:]: #exclude 0
                truncated_text_ids, trunc_text_words, trunc_seg_ids = self._truncate_words(sorted_idx, text_words, text_ids, num, seg_ids=seg_ids)
                trunc_class, trunc_prob = self.predict(truncated_text_ids, target, seg_ids=trunc_seg_ids)
                instance_degradation_prob.append(trunc_prob)
                instance_degradation_acc.append(trunc_class==target)
            self.degradation_prob.append(np.array(instance_degradation_prob))
            self.degradation_acc.append(np.array(instance_degradation_acc))
    
    def eval_replace(self, attribution, text_words, text_ids, target, mask, mask_id, seg_ids=None):
        trunc_ids = []
        total_len = len(text_words)
        granularity = np.linspace(0, 1, self.step)
        trunc_words_num = [int(g) for g in np.round(granularity*total_len)]
        trunc_words_num = list(dict.fromkeys(trunc_words_num))
        # sort idx from high importance to low importance by default
        sorted_idx = np.argsort(-attribution)
        if self.reverse:
            sorted_idx = np.argsort(attribution)

        with torch.no_grad():
            original_class, original_prob = self.predict(text_ids, target)
            self.original_prob.append(original_prob)
            self.original_acc.append(original_class==target)

            instance_degradation_prob = []
            instance_degradation_acc = []
            for num in trunc_words_num[1:]: #exclude 0
                truncated_text_ids, trunc_text_words = self._replace_words(sorted_idx, text_words, text_ids, num, mask, mask_id)
                trunc_class, trunc_prob = self.predict(truncated_text_ids, target, seg_ids=seg_ids)
                instance_degradation_prob.append(trunc_prob)
                instance_degradation_acc.append(trunc_class==target)
            self.degradation_prob.append(instance_degradation_prob)
            self.degradation_acc.append(instance_degradation_acc)

    def return_degradation_result(self):
        return self.degradation_prob, self.degradation_acc

    def return_original_result(self):
        return self.original_prob, self.original_acc