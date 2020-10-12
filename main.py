import os
import numpy as np
import argparse
from code.evaluations.Evaluation import *
from code.utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="evaluations different interpretable methods")
    parser.add_argument('--method', default='iba',
                        help='interpretable method to evaluate, choices are: `iba`')
    parser.add_argument('--dataset', default='imdb',
                        help='dataset to evaluate, choices are: `imdb`, `mnli`, `rte`, `agnews`')
    parser.add_argument('--data_path', default='../data/imdb', help='data path, can be a directory or filename')
    parser.add_argument('--model_state_dict_path', default='finetuned_model/',
                        help='path stored fine tuned torch state dict')
    parser.add_argument('--model_pickled_dir', default=None, help='directory stored pickled fine tuned torch model')
    parser.add_argument('--layer_idx', default=8, help="layer idx to insert with iba") # corresponds to the actual layer index - 1
    parser.add_argument('--test_sample', default=1000, help="test sample number, you can simply input `all` if you want to evaluate the whole dataset")
    parser.add_argument('--degrad_step', default=10, help="degradation steps")
    parser.add_argument('--reverse', default=False, action="store_true", help="reverse the importance attribution")
    parser.add_argument('--replace', default=False, action="store_true", help="replace target with mask")
    parser.add_argument('--beta', default=1e-5, help="beta for iba")
    parser.add_argument('--train_steps', default=10, help="training steps for iba")
    parser.add_argument('--lr', default=1, help='learning rate for iba')
    parser.add_argument('--output_path', default=None, help='if not None, save the result numpy array to that path')

    args = parser.parse_args()

    method = args.method
    layer_idx = int(args.layer_idx)
    dataset = args.dataset
    data_path = args.data_path
    model_state_dict_path = args.model_state_dict_path
    model_pickled_dir = args.model_pickled_dir
    degrad_step = int(args.degrad_step)
    reverse = args.reverse
    replace = args.replace
    beta = float(args.beta)
    train_steps = int(args.train_steps)
    lr = float(args.lr)
    output_path = args.output_path
    task = TASK_DICT[dataset]
    class_num = int(CLASS_DICT[dataset])  # for LIME evaluations
    special_tokens = {"[CLS]", "[SEP]"}  # special tokens we exclude for degradation test
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load data
    load_func = 'process_{}'.format(dataset)
    test_data = eval(load_func)(data_path)
    test_samples_number = int(args.test_sample) if args.test_sample!='all' else len(test_data)

    # Start evaluations!
    original_acc, original_prob, degrad_acc, degrad_prob = evaluation_degradation(model_state_dict_path,
                                                                                  model_pickled_dir,
                                                                                  method,
                                                                                  task,
                                                                                  special_tokens,
                                                                                  test_data,
                                                                                  test_samples_number,
                                                                                  class_num,
                                                                                  degrad_step,
                                                                                  layer_idx,
                                                                                  reverse,
                                                                                  device,
                                                                                  beta,
                                                                                  lr,
                                                                                  train_steps)

    # Check if need to save
    if output_path:
        np.save(os.path.join(output_path, 'original_prob_{}_{}'.format(method, dataset)), original_prob)
        np.save(os.path.join(output_path, 'original_acc_{}_{}'.format(method, dataset)), original_acc)
        np.save(os.path.join(output_path, 'degrad_prob_{}_{}'.format(method, dataset)), degrad_prob)
        np.save(os.path.join(output_path, 'degrade_acc_{}_{}'.format(method, dataset)), degrad_acc)
