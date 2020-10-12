## Code for Paper: Inserting Information Bottleneck for Attribution in Transformers

Below is the instruction of using IBA to visualize the attribution map for transformer, especially BERT.  

### Environment Setup
* Create virtual environment and install packages, below is the example of using conda
    * `conda create -n iba python=3.7`
    * `source activate iba`
    * `pip install -r requirements.txt`
* Clone the repo
    * `git clone https://github.com/bazingagin/IBA.git`

### Example Usage
You can use either pretrained or fine-tuned model with your own choice of sentence to visualize the attribution map.
* Start jupyter notebook to view IBA-tutorial.ipynb
    * `cd IBA && jupyter notebook IBA-tutorial.ipynb`    


### Evaluation
* Download dataset
    * IMDB: <https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz>
    * AGNews: <https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv> 
    * MNLI, RTE: <https://gluebenchmark.com/tasks>
* Download fine-tuned model
    * `wget https://storage.googleapis.com/iba4nlp-saved-model/finetuned_model.zip`
    * `imdb` and `agnews` are pytorch state_dict while `RTE` and `MNLI` are pytorch pickled model
* Set data directory and model directory for each dataset. For example,
    * `export IMDB_DATA_PATH=data/aclImdb/test`
    * `export IMDB_FINETUNED_MODEL=model/imdb.model` 
* Run! Below is the example of evaluation on 10 examples inserting IB after layer 9. You can use `all` for `test_sample` to evaluate on the whole dataset.
    * ```python main.py --data_path ${IMDB_DATA_PATH} --dataset imdb --layer_idx 8 --model_state_dict_path ${IMDB_FINETUNED_MODEL} --test_sample 10```
    
*Note:* 
1. You can either use pytorch's state_dict, or the entire model. Just specify with different arguments - `model_state_dict_path` or `model_pickled_dir`. 
2. `layer_idx` is 0-indexed, so you can choose from 0 to 11 instead of 1 to 12.
 

### Acknowledgement
* Thanks to Schulz et al. 2020 for providing IBA code base. If you are interested in using IBA for CV, you can refer to their code base [here](https://github.com/BioroboticsLab/IBA-paper-code.git)

### Citation

