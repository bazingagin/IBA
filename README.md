# Code for Paper: Inserting Information Bottleneck for Attribution in Transformers

This paper is accepted to [EMNLP2020 Findings](https://www.aclweb.org/anthology/2020.findings-emnlp.343/).  
Below is the instruction of using IBA to visualize the attribution map for transformer, especially BERT.  

## Environment Setup
1. Create virtual environment and install packages, below is the example of using conda
```
conda create -n iba python=3.7
source activate iba
pip install -r requirements.txt
```
2. Clone the repo
```
git clone https://github.com/bazingagin/IBA.git
```

## Example Usage
You can use either pretrained or fine-tuned model with your own choice of sentence to visualize the attribution map.
1. Start jupyter notebook to view IBA-tutorial.ipynb
```
cd IBA && jupyter notebook IBA-tutorial.ipynb
```    


## Evaluation
0. Create directory
```
mkdir data && cd data
```
1. Download dataset
    * IMDB: <https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz>
    * AGNews: <https://github.com/mhjabreel/CharCnn_Keras/tree/master/data/ag_news_csv> 
    * MNLI, RTE: <https://gluebenchmark.com/tasks>
    
    and untar archive file if necessary:
      * `tar -xvf aclImdb_v1.tar.gz`
2. Download fine-tuned model
```
wget https://storage.googleapis.com/iba4nlp-saved-model/finetuned_model.zip
unzip finetuned_model.zip
```
`imdb` and `agnews` are pytorch state_dict while `RTE` and `MNLI` are pytorch pickled model

3. Set data directory and model directory for each dataset. For example,
```
export IMDB_DATA_PATH=data/aclImdb/test
export IMDB_FINETUNED_MODEL=finetuned_model/imdb.model
```
4. Run! Below is the example of evaluation on 10 examples inserting IB after layer 9. You can use `all` for `test_sample` to evaluate on the whole dataset.
```
python main.py --data_path ${IMDB_DATA_PATH} 
               --dataset imdb 
               --layer_idx 8 
               --model_state_dict_path ${IMDB_FINETUNED_MODEL} 
               --test_sample 10
```
    
*Note:* 
1. You can either use pytorch's state_dict, or the entire model. Just specify with different arguments - `model_state_dict_path` or `model_pickled_dir`. 
2. `layer_idx` is 0-indexed, so you can choose from 0 to 11 instead of 1 to 12.
 

## Acknowledgement
* Thanks to Schulz et al. 2020 for providing IBA code base. If you are interested in using IBA for CV, you can refer to their code base [here](https://github.com/BioroboticsLab/IBA-paper-code.git)

## Citation

* Please consider cite the paper if you've found code useful :)

```
@inproceedings{jiang-etal-2020-inserting,
    title = "{I}nserting {I}nformation {B}ottlenecks for {A}ttribution in {T}ransformers",
    author = "Jiang, Zhiying  and
      Tang, Raphael  and
      Xin, Ji  and
      Lin, Jimmy",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.343",
    pages = "3850--3857",
    abstract = "Pretrained transformers achieve the state of the art across tasks in natural language processing, motivating researchers to investigate their inner mechanisms. One common direction is to understand what features are important for prediction. In this paper, we apply information bottlenecks to analyze the attribution of each feature for prediction on a black-box model. We use BERT as the example and evaluate our approach both quantitatively and qualitatively. We show the effectiveness of our method in terms of attribution and the ability to provide insight into how information flows through layers. We demonstrate that our technique outperforms two competitive methods in degradation tests on four datasets. Code is available at \url{https://github.com/bazingagin/IBA}.",
}
```
