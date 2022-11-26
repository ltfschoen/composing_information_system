# Setup

## Pipeline for Question & Answer

### About

* Given user input query
* Search info in dataset by pipeline
* Answer returned containing relation, sentence

### Setup Q&A NLP Pipeline

#### Install PyEnv
* Clone repo and switch into project root folder
* Install pyenv
```
brew install pyenv
eval "$(pyenv init -)"
```

* Run the following to add to ~/.bash_profile https://github.com/pyenv/pyenv#set-up-your-shell-environment-for-pyenv
```
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile
echo 'eval "$(pyenv init -)"' >> ~/.bash_profile
```

```
source ~/.bash_profile
```

#### Update Python Path

* Change to root project directory
* Update Python path, so it can find Python modules

```
export PYTHONPATH="$(PWD):$PYTHONPATH"
echo $PYTHONPATH
```

#### Install Python 3

* Install Python 3
```
pyenv install -l
pyenv install 3.7.13
```

#### Install Dependencies

* Add dependencies. See .python-version for compatible version
```
pyenv local 3.7.13
pyenv shell 3.7.13
python --version
pip install Cython
pip install -r requirements.txt
```

    * Note: 
        It may be necessary to clone the forte-wrappers repo https://github.com/asyml/forte-wrappers, then switch to `pyenv local 3.7.13`, then install the dependencies from source with
        ```
        pip install src/nltk src/elastic src/allennlp src/spacy
        ```

#### Install and Run ElasticSearch

* Install ElasticSearch from https://www.elastic.co/guide/en/elasticsearch/reference/current/targz.html

```
curl -O https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.11.1-darwin-x86_64.tar.gz
curl https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-7.11.1-darwin-x86_64.tar.gz.sha512 | shasum -a 512 -c - 
tar -xzf elasticsearch-7.11.1-darwin-x86_64.tar.gz
```

* Move
    ```
    echo 'export PATH="/Users/cuttlefish/Downloads/elasticsearch-7.11.1/bin:$PATH"' >> ~/.bash_profile
    source ~/.bash_profile
    ```

* Run ElasticSearch server on backend in another terminal

```
elasticsearch
```

#### Build Elasticsearch Indexer

* Build Elasticsearch indexer.
    * Dataset: CORD-19 dataset we used the data in json format for index, which is in `data/document_parses/????`. Sample data folder for testing code which is located in `data/document_parses/sample_pdf_json`.
    * Indexer config: Change the config of ElasticSearch in `examples/pipeline/indexer/config.yml`
    
* Run ElasticSearch indexer to index the files in `your_data_directory` (i.e. ./data/document_parses/sample_pdf_json)

```
python examples/pipeline/indexer/cordindexer.py --data-dir [your_data_directory]
```
    
* Expect output

```
INFO:elasticsearch:GET http://localhost:9200/ [status:200 request:0.057s]
INFO:elasticsearch:POST http://localhost:9200/_bulk?refresh=true [status:200 request:0.336s]
```

#### Build QA engine

Start QA pipeline in the command line using:
* Scispacy models for entity linking
* AllenNLP models for SRL and OpenIE task.

* Run the following to initialize the pipeline

```
python examples/pipeline/inference/search_cord19.py
```

* Errors because CUDA is not supported with Mac, because Macs typically don't have nvidia GPUs. Might have to install similar to this https://discuss.pytorch.org/t/installing-pytorch-on-a-machine-without-gpu/70286.
Raised an issue https://github.com/petuum/composing_information_system/issues/27

    ```
    Installing collected packages: en-ner-jnlpba-md
    Successfully installed en-ner-jnlpba-md-0.3.0
    Traceback (most recent call last):
    File "examples/pipeline/inference/search_cord19.py", line 90, in <module>
        nlp.initialize()
    File "/Users/cuttlefish/.pyenv/versions/3.7.13/lib/python3.7/site-packages/forte/pipeline.py", line 710, in initialize
        self.initialize_components()
        ...
    File "/Users/cuttlefish/.pyenv/versions/3.7.13/lib/python3.7/site-packages/torch/cuda/__init__.py", line 166, in _lazy_init
        raise AssertionError("Torch not compiled with CUDA enabled")
    AssertionError: Torch not compiled with CUDA enabled
    ```

    * Solution attempts:
        * Run `export CUDA_VISIBLE_DEVICES=""` before running  `python examples/pipeline/inference/search_cord19.py`
        * See other alternatives here https://stackoverflow.com/questions/53266350/how-to-tell-pytorch-to-not-use-the-gpu
        * If using conda then could use `cpuonly -c` flag

* `RuntimeError: Found no NVIDIA driver on your system. Please check that you have an NVIDIA GPU and install`
    * Run `export CUDA_VISIBLE_DEVICES=""` before running  `python examples/pipeline/inference/search_cord19.py`

When you see `Enter your query here:`, you can start to ask questions, such as the following and the pipeline will process the query, then search the relations and output the result to human readable format.
```
'what does covid-19 cause', 
'what does covid-19 affect', 
'what caused liver injury', 
'what caused renal involvement'
```

Here is an example of the question 'what does covid-19 cause'. 

Output shows the potential relation, the source sentence and source article that the relation comes from, 
as well as the relevant UMLS medical concepts in the relations.

```
•Relation:
    COVID-19	causes	infection in the pulmonary system

•Source Sentence:
    COVID-19 enters the cell by penetrating ACE2 at low cytosolic pH and causes infection in the pulmonary system [2, 3] .(From Paper: , Can COVID-19 cause myalgia with a completely different mechanism? A hypothesis)

•UMLS Concepts:
 - covid-19
	Name: COVID-19	CUI: C5203670	Learn more at: https://www.ncbi.nlm.nih.gov/search/all/?term=C5203670
============================
```

* Output

```
...

For maximum performance, you can install NMSLIB from sources 
pip install --no-binary :all: nmslib
```

#### Pipeline Introduction

The pipeline contains three major steps: Query Understanding, Document Retrieval, and Answer Extraction.

##### Query Understanding
Forte analyzes user’s input question by annotating the basic language features using __NLTK__ word tokenizer, POS tagger and lemmatizer. 
Then __AllenNLP’s SRE model__ was utilized to extract the arguments and predicate in the question, and which argument that the user is interested in. 
This annotated question is transformed into a query and pass to the next step. 

> Note: This step is expected to extract the user expected relation and it is relied on AllenNLP's SRE result.
So the question has to contain two arguments and one predicate successfully caught by the model, otherwise this step will fail.
You can check the section above to see the pattern of sample queries.

##### Document Retrieval
The __ElasticSearch__ backend is fast and quickly filters information from a large corpus, 
which is great for larger corpora with millions of documents. We use it as search engine here. 

The query created in the last step was used to retrieve relevant articles from an index that’s stored in ElasticSearch. 

The extracted documents was stored as datapack in Forte, and passed to the next step to generate final output.

> Note: You can set the search configs in `indexer/config.yml`, for example, `query_creator.size` controls the number of retrieved documents, `query_creator.query_pack_name` will set the name of query datapack in pipeline. 
But you have to keep `indexer.index_config` consistent with Create Index step to make the search work.

##### Answer Extraction
Given relevant document datapacks, the system helps to extract the relevant relations. 

Here, __ScispaCy models__ trained on biomedical text were utilized to do __sentence parsing__, __NER__, and __entity linking__ with UMLS concepts. 
__AllenNLP’s OpenIE model__ was utilized for __relation extraction__. 
__NLTK__ Lemmatizer was also used to process predicate of the relations.

Given all relations, the relation that matched user's query will be selected as candidate answer, which means the predicate lemma in the relation and user query's predicate lemma are the same, and the user query's argument was mentioned in the extracted relation.

Finally, the relations were polished by adding references and supporting information that’s useful for fact-checking and countering misinformation.
So the source sentence and article were also provided in the result. 
Besides, some terms in the relation could be linked to UMLS concepts, which brings standards for further interoperability or research, so they were also listed on the result.

The final result was organized to three parts: __Relation, Source Sentence, and UMLS Concepts__ for reading purpose. 

## Pipeline for Training

### About

* Given answer returned containing relation, sentence
* Entity linking, in which you will be able to see how to do a composable training and inference pipeline with your own model in Forte.
    * Using training pipelines: NER, and entity linking

### Tasks and Datasets

**General NER**

* Task: NER in general domain.

* Model: General BERT.

* Dataset for model training :   
    English data from CoNLL 2003 shared task. It contains four different types of named entities: PERSON, LOCATION, ORGANIZATION, and MISC.
    It can be downloaded from [DeepAI website](https://deepai.org/dataset/conll-2003-english).

* Use Case:  ​Analyzing research papers on COVID-19​

    * Materials to use for testing:  ​Research Papaers for COVID-19.​

    * Model: BioBERT v1.1 (domain-specific language representation model pre-trained on large-scale biomedical corpora)​

    * Dataset for model training: 
    NER: CORD-NER dataset,  Entity-Linking: CORD-NERD Dataset​
​

**Bio-medical NER**

* Task: NER in bio-medical domain.

* Model: BioBERT v1.1 (domain-specific language representation model pre-trained on large-scale biomedical corpora).

* Dataset for model training:   
    MTL-Bioinformatics-2016. Download and know more about this dataset on 
    [this github repo](https://github.com/cambridgeltl/MTL-Bioinformatics-2016). 


**Wiki entity linking**

* Task: Entity linking in general domain.

* Model: General Bert.

* Dataset for model training: 
    AIDA CoNLL03 entity linking dataset. The entities are identified by YAGO2 entity name, by Wikipedia URL, or by Freebase mid.
    It has to be used together with CoNLL03 NER dataset mentioned above. 
    
    First download CoNLL03 dataset which contains train/dev/test.
    
    Second, download aida-yago2-dataset.zip from [this website](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/ambiverse-nlu/aida/downloads).
    
    Third, in the downloaded folder, manually segment AIDA-YAGO2-annotations.tsv into three files corresponding to CoNLL03 train/dev/test,
    then put them into train/dev/test folders.


**Medical entity linking**

* Task: Entity linking in medical domain.

* Model: BioBERT v1.1 (domain-specific language representation model pre-trained on large-scale biomedical corpora).

* Dataset for model training: 
    MedMentions st21pv sub-dataset. It can be download from [this github repo](https://github.com/chanzuckerberg/MedMentions/tree/master/st21pv).


### How to Train Models

* Below shows the steps to train a general NER model. You can modify config directory to train others.

* Create a conda virtual environment and git clone this repo, then in command line:

```
cd composing_information_system/
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

* Create an output directory.

```
mkdir sample_output
```

* Run training script:

```
python examples/tagging/main_train_tagging.py --config-dir examples/tagging/configs_ner/
```

* After training, you will find your trained models in the following directory. It contains the trained model, vocabulary, train state and training log. 

```
ls sample_output/ner/
```

    

### How to do Inference

* Download the pre-trained models and vocabs from below list. Put model.pt and vocab.pkl into `predict_path` specified in `config_predict.yml`.
Then run the following command.

```
python examples/tagging/main_predict_tagging.py --config-dir examples/tagging/configs_ner/
```

    + General Ner : [__model__](https://drive.google.com/file/d/1WCSwDw8WEjshf1IUY4iMPQBV_HyhuwNm/view?usp=sharing), 
    [__train_state__](https://drive.google.com/file/d/1cDmDNFDZLgZLr2BO4MeuMjD_eh4T3PQI/view?usp=sharing), 
    [__vocab__](https://drive.google.com/file/d/18UFHFg9gfZbb9Sb5s8_h0eG2sd9jXn4H/view?usp=sharing)

    + Bio-medical NER : [__model__](https://drive.google.com/file/d/1dL2PYSPb-HOiSBQeQiVD530uxmLGbfbP/view?usp=sharing), 
    [__train_state__](https://drive.google.com/file/d/1bJq1RUGK1h3epjklFZEMGLY34SEO9Svh/view?usp=sharing), 
    [__vocab__](https://drive.google.com/file/d/1yhQriZjABv3XA_0I4w9jD8k3n3SFvJqc/view?usp=sharing)

    + Wiki entity linking : [__model__](https://drive.google.com/file/d/1injnv7s8a-PAhfwMN25kyzxh-FGwnWNZ/view?usp=sharing), 
    [__train_state__](https://drive.google.com/file/d/1pttk34Fk3fWJz-Vfy3kCY8ET7qReJH84/view?usp=sharing), 
    [__vocab__](https`://drive.google.com/file/d/19OVGetDQ7BJ1m1_FyxkDE2SI266XvNLv/view?usp=sharing)
    
    + Medical entity linking : [__model__](https://drive.google.com/file/d/1kBDItWrguZez0F57xmT90eHEucc2CjE-/view?usp=sharing), 
    [__train_state__](https://drive.google.com/file/d/1qbL7gb3SMvgHUMhuD_-tQFFvXYBvV6Pl/view?usp=sharing), 
    [__vocab__](https://drive.google.com/file/d/1UrcIF3ZwllWdee0wEbCYpZ6x9bGXwEdy/view?usp=sharing)



### How to do Inferencing using two concatenated Models

* You can use your trained bio-medical NER model and medical entity linking model to do inference on a new dataset

* Inference Dataset :
    CORD-NERD dataset. Information about this dataset and downloadble links can be found [here](https://aistairc.github.io/BENNERD/).

    `python examples/tagging/main_predict_cord.py --config-dir examples/tagging/configs_cord_test/`


### Evaluation of Performance examples:

* Below is the performance metrics of the General NER task.

    
    |       General NER          |  Accuracy | Precision | Recall | F1    | #instance |
    |----------------------------|-----------|-----------|--------|-------|-----------|
    |   Overall                  |   98.98   | 93.56     | 94.81  | 94.18 |           |
    |   LOC                      |           | 95.94     | 96.41  | 96.17 |  18430    |
    |   MISC                     |           | 86.15     | 89.97  | 88.02 |   9628    |
    |   ORG                      |           | 91.05     | 91.99  | 91.52 |  13549    |
    |   PER                      |           | 96.89     | 97.69  | 97.29 |  18563    |



* Below is the performance metrics of the bio-medical NER task.

    
    | Bio-medical NER   |  Accuracy  | Precision| Recall | F1    |  #instance |
    |-------------------|------------|----------|--------|-------|------------|
    |   Overall         |   98.41    | 84.93    | 89.01  | 86.92 |            |
    |   Chemical        |            | 79.20    | 86.34  | 82.62 | 1428       |
    |   Organism        |            | 85.23    | 73.87  | 79.14 | 3337       |
    |   Protein         |            | 85.53    | 97.15  | 90.97 | 11972      |
  

* Below is the performance metrics of the wiki entity linking task. 
Due to the large number of classes in entity linking tasks, we are only showing the overall performance.


    |   Wiki entity linking      |  Accuracy | Precision | Recall | F1    | 
    |----------------------------|-----------|-----------|--------|-------|
    |        Overall             |   91.27   |  51.86    | 38.60  | 44.25 |


* Below is the performance metrics of the medical entity linking task. 
Since MedMentions dataset does not provide word boundaries (only has entity linking boundaries), the evaluation method here is to count exact match of entities.

    |   Medical entity linking        |  Precision | Recall    | F1     | 
    |---------------------------------|------------|-----------|--------|
    |       Exact match               |   26.25    | 22.24     | 24.07  |
