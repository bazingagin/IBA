from scripts.information_bottleneck.IBA import *

np.random.seed(3)


# Feature Map is the output of a certain layer given X
def extract_feature_map(model, layer_idx, text_ids, seg_ids=None):
    with torch.no_grad():
        state = model(text_ids, token_type_ids=seg_ids)
        feature = state[1][layer_idx]
        return feature


# Extract BERT Layer
def extract_bert_layer(model, layer_idx):
    desired_layer = ''
    for name, submodule in model.named_children():
        for n, s in submodule.named_children():
            if n == 'encoder':
                for n2, s2 in s.named_children():
                    for n3, s3 in s2.named_children():
                        if n3 == str(layer_idx):
                            desired_layer = s3
                            return desired_layer


def get_original_representation(model, text_ids):
    with torch.no_grad():
        state = model(text_ids)
        features = state[1]
    return features  # length of 13 tuple


# Initialize Interpretability Reader
def initialize_reader(model, desired_layer, feature_map, beta, lr, train_steps):
    estimator = Estimator(desired_layer)
    estimator.feed(feature_map.cpu().numpy())
    reader = IBAInterpreter(model, estimator, beta=beta, lr=lr, steps=train_steps, progbar=True)
    return reader


# Layer Heatmap
def layer_heatmap_iba(text1, text2, target, text_words, text_ids, segment_ids, model, layer_idx, beta, lr, train_steps):
    features = extract_feature_map(model, layer_idx, text_ids, segment_ids)
    layer = extract_bert_layer(model, layer_idx)
    reader = initialize_reader(model, layer, features, beta, lr, train_steps)
    heatmap = reader.bert_heatmap(text_ids, target, segment_ids)
    return heatmap
