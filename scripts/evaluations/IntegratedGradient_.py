import torch
import torch.nn as nn


def construct_input_ref_pair(tokenizer, device, text, ref_token_id, sep_token_id, cls_token_id):
    text_ids = tokenizer.encode(text, add_special_tokens=False, max_length=510)
    input_ids = [cls_token_id] + text_ids + [sep_token_id]
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(text_ids) + [sep_token_id]
    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device)


def construct_input_ref_pair_et(tokenizer, device, question, text, ref_token_id, sep_token_id, cls_token_id):
    question_ids = tokenizer.encode(question, add_special_tokens=False, max_length=250)
    text_ids = tokenizer.encode(text, add_special_tokens=False, max_length=250)

    # construct input token ids
    input_ids = [cls_token_id] + question_ids + [sep_token_id] + text_ids + [sep_token_id]

    # construct reference token ids 
    ref_input_ids = [cls_token_id] + [ref_token_id] * len(question_ids) + [sep_token_id] + \
        [ref_token_id] * len(text_ids) + [sep_token_id]

    return torch.tensor([input_ids], device=device), torch.tensor([ref_input_ids], device=device), len(question_ids)


def construct_input_ref_token_type_pair(input_ids, device, sep_ind=0):
    seq_len = input_ids.size(1)
    token_type_ids = torch.tensor([[0 if i <= sep_ind else 1 for i in range(seq_len)]], device=device)
    ref_token_type_ids = torch.zeros_like(token_type_ids, device=device)# * -1
    return token_type_ids, ref_token_type_ids


def compute_bert_outputs(model_bert, embedding_output, attention_mask=None, head_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones(embedding_output.shape[0], embedding_output.shape[1]).to(embedding_output)

    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

    extended_attention_mask = extended_attention_mask.to(dtype=next(model_bert.parameters()).dtype) # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

    if head_mask is not None:
        if head_mask.dim() == 1:
            head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
            head_mask = head_mask.expand(model_bert.config.num_hidden_layers, -1, -1, -1, -1)
        elif head_mask.dim() == 2:
            head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
        head_mask = head_mask.to(dtype=next(model_bert.parameters()).dtype) # switch to fload if need + fp16 compatibility
    else:
        head_mask = [None] * model_bert.config.num_hidden_layers

    encoder_outputs = model_bert.encoder(embedding_output,
                                         extended_attention_mask,
                                         head_mask=head_mask)
    sequence_output = encoder_outputs[0]
    pooled_output = model_bert.pooler(sequence_output)
    outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
    return outputs  # sequence_output, pooled_output, (hidden_states), (attentions) 


def interpret_sentence(model_wrapper, tokenizer, device, interpret_alg, sentence1, sentence2, label=1):

    model_wrapper.eval()
    model_wrapper.zero_grad()
    
    ref_token_id = tokenizer.pad_token_id
    sep_token_id = tokenizer.sep_token_id 
    cls_token_id = tokenizer.cls_token_id
    if sentence2 is None:
        input_ids, ref_input_ids = construct_input_ref_pair(tokenizer, device, sentence1, ref_token_id, sep_token_id, cls_token_id)
        token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, device)
        input_embedding = model_wrapper.model.bert.embeddings(input_ids)
        ref_embedding = model_wrapper.model.bert.embeddings(ref_input_ids)
    else:
        input_ids, ref_input_ids, sep_ids = construct_input_ref_pair_et(tokenizer, device, sentence1, sentence2, ref_token_id, sep_token_id, cls_token_id)
        token_type_ids, ref_token_type_ids = construct_input_ref_token_type_pair(input_ids, device, sep_ids)
        input_embedding = model_wrapper.model.bert.embeddings(input_ids, token_type_ids=token_type_ids)
        ref_embedding = model_wrapper.model.bert.embeddings(ref_input_ids, token_type_ids=token_type_ids)

    
    # predict
    pred = model_wrapper(input_embedding).item()
    pred_ind = round(pred)

    # compute attributions and approximation delta using integrated gradients
    attributions_ig, delta = interpret_alg.attribute(input_embedding, baselines=ref_embedding, n_steps=10, return_convergence_delta=True)
    attributions = attributions_ig.sum(dim=2).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()

    return attributions


class BertModelWrapper(nn.Module):
    
    def __init__(self, model):
        super(BertModelWrapper, self).__init__()
        self.model = model
        
    def forward(self, embeddings):        
        outputs = compute_bert_outputs(self.model.bert, embeddings)
        pooled_output = outputs[1]
        pooled_output = self.model.dropout(pooled_output)
        logits = self.model.classifier(pooled_output)
        return torch.softmax(logits, dim=1)[:, 0].unsqueeze(1) # calculate attribution towards positive

