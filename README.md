# EDGAR10-Q Dataset 

This dataset is built from 10-Q/K documents (Quarterly and Yearly Reports) of publicly listed companies on the SEC. To access these documents, follow [this link](https://www.sec.gov/os/accessing-edgar-data). Please see sample.csv to find the instance of a document of the dataset. To get CIK of an organization, use the CIK_lookup in contents folder. 

You may want to check out 
* Our paper: [CONTEXT-NER: Contextual Phrase Generation at Scale](https://arxiv.org/abs/2109.08079/)



## Dataset Access

The dataset is freely available to use on huggingface as [EDGAR10-Q dataset](https://huggingface.co/datasets/him1411/EDGAR10-Q)

Here is how to access it: 

'''
from datasets import load_dataset
dataset = load_dataset("him1411/EDGAR10-Q")
'''


### Data Fields

The data fields are the same among all splits.

- `text`: a `string` in the form of entity plus sentence. 
- `label`: a string describing the relevant context for entity in the sentence

### Data Splits

The dataset is split into train, validation, and test sets. The sizes of the splits are as follows:

|           | Train     | Validation | Test  |
|-----------|-----------|------------|-------|
| Instances | 1,498,995 | 187,383    |187,383|



## Supervised Trained Models

6 sequence to sequence models were finetuned/ instruction tuned using the train split of the dataset. They are: 

1. [EDGAR-T5-base](https://huggingface.co/him1411/EDGAR-T5-base)
2. [EDGAR-BART-Base](https://huggingface.co/him1411/EDGAR-BART-Base)
3. [EDGAR-flan-t5-base](https://huggingface.co/him1411/EDGAR-flan-t5-base)
4. [EDGAR-T5-Large](https://huggingface.co/him1411/EDGAR-T5-Large) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zero-shot-open-information-extraction-using/contextner-on-edgar10-q-dataset)](https://paperswithcode.com/sota/contextner-on-edgar10-q-dataset?p=zero-shot-open-information-extraction-using)
5. [EDGAR-Tk-Instruct-Large](https://huggingface.co/him1411/EDGAR-Tk-Instruct-Large)
6. [Instruction tuned EDGAR-Tk-Instruct-base](https://huggingface.co/him1411/EDGAR-Tk-instruct-base-inst-tune)

Please checkout the hugging face model cards on how to use them. 



## Building Dataset

Using the script dataset_generation_and_baseline.py will pull the the data from sec website and store it in content folder. cik_lookup.xlsx has the list of 2000 organizations whose data was pulled. The script will also run the baseline approach and store all the results in each organizations' excel respectively.


## ChatGPT response generation
Once the dataset is created and baseline appraoch is executed and the excel is complete, use the script chatgpt_responses.py for getting the reuslts from ChatGPT. Please use your own API key for its execution.


Table 1 :  Instance of the Dataset.
|                                                                        Sentence                                                                       |    value    | entity type |                          Labels for each entity                          |
|:------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------:|:-----------:|:------------------------------------------------------------------------:|
| As of August 5, 2019, there were 46,662,179 shares of common stock, $0.01 par value, outstanding.                                                      | 4,66,62,179 | CARDINAL    | Shares Outstanding                               |
| The Company also derecognized existing deferred rent liabilities of $15,302.                                                                           | 15,302      | MONEY       | Rent Expense                                        |
| The intangible assets acquired have a weighted average useful life of approximately nine years.                                                        | nine years  | DATE        | Intangible assets |
| The initial purchase price of $31,676 included $30,176 cash consideration paid upon acquisition,  funded primarily through borrowings under the Senior | 31,676      | MONEY       | Initial purchase price |
| Credit Facility, and a contingent earn out payment of up to $25,000 with an estimated fair  value of $1,500 as of the acquisition date.                | 30,176      | MONEY       | Payments to  Acquire Businesses                      |


Table 2: Statistics about dataset: 

![Imgur](https://i.imgur.com/zlXq2Cp.png)




## Results for baseline algorithm, ChatGPT responses and supervised learning models

![Imgur](https://i.imgur.com/sR4zFvt.png)

## Supervised Finetuning
Use supervised_deepspeed_finetuning.sh for finetuning any model on EDGAR10-Q dataset.


## Results on Dowstream datasets
[EDGAR-T5-Large](https://huggingface.co/him1411/EDGAR-T5-Large) was finetuned on some downstream datasets to get better results than T5 large. BloombergGPT 50B was used as baseline. 

| Dataset  | Bloomberg GPT 50B | T5 Large | Edgar T5 Large |
|----------|-------------------|----------|----------------|
| FiQA SA  | 75.07             | 74.89    | 80.42          |
| FPB      | 51.07             | 55.77    | 79.69          |
| Headline | 82.20             | 90.55    | 93.55          |


BibTeX Entry and Citation Info
===============
If you are using our model, please cite our paper:

```bibtex
@article{gupta2021context,
  title={Context-NER: Contextual Phrase Generation at Scale},
  author={Gupta, Himanshu and Verma, Shreyas and Kumar, Tarun and Mishra, Swaroop and Agrawal, Tamanna and Badugu, Amogh and Bhatt, Himanshu Sharad},
  journal={arXiv preprint arXiv:2109.08079},
  year={2021}
}
```

