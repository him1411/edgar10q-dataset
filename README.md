# EDGAR10-Q Dataset [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/zero-shot-open-information-extraction-using/contextner-on-edgar10-q-dataset)](https://paperswithcode.com/sota/contextner-on-edgar10-q-dataset?p=zero-shot-open-information-extraction-using)

This dataset is built from 10-Q/K documents (Quarterly and Yearly Reports) of publicly listed companies on the SEC. To access these documents, follow [this link](https://www.sec.gov/os/accessing-edgar-data). Please see sample.csv to find the instance of a document of the dataset. To get CIK of an organization, use the CIK_lookup in contents folder. 


## Dataset

The dataset is available via [this link](https://arizonastateu-my.sharepoint.com/:f:/g/personal/hgupta35_sundevils_asu_edu/Eo0a8sp7YYNLo1Wkfae0_Q4B6SXPzsUjBjd4b1HsnWCbIQ?e=q7Qxjw)


## Building Dataset

Using the script dataset_generation_and_baseline.py will pull the the data from sec website and store it in content folder. cik_lookup.xlsx has the list of 2000 organizations whose data was pulled. The script will also run the baseline approach and store all the results in each organizations' excel respectively.


## GPT-3 response generation
Once the dataset is created and baseline appraoch is executed and the excel is complete, use the script gpt3_response_generation.py for getting the reuslts from GPT-3. Please use your own API key for its execution.


Some Statistics about the corpus are given below : 

Table 1 : Basic Statistics of the dataset

| Statistics about the corpus             | Numbers |
|-----------------------------------------|---------|
| Total  # of Documents in Corpus         | 18752   |
| Number of Sentences                     | 1009712 |
| Total Words in corpus                   | 77400425|
| Total Entities                          | 2780969 |
| Total Labels                            | 4032405 |


Table 2 : Sentence Wise Statistics of the dataset


| Sentences with # of Entities | # of Sentences | Average Word Length of the Sentence |
|------------------------------|---------------:|-------------------------------------|
| 1 Entity                     | 503433         | 31.70                               |
| 2 Entity                     | 331130         | 36.30                               |
| 3 Entity                     | 108103         | 42.40                               |
| 4 Entity                     | 47191          | 52.80                               |
| 5 Entities and above         | 19855          | 73.90                               |


Table 3 : Statistics of the entities in the dataset

|    Entity Types                       |  Counts |
|--------------------------------------:|:-------:|
| Floating Values (Monetary & percent)  | 2143054 |
| number of assests (Shares & Integers) | 425850  |
| Ordinal Values                        | 16891   |
| Dates                                 | 195174  |

Table 4 :  Instance of the Dataset.


|                                                                        Paragraph                                                                       |    value    | entity type |                          Labels for each entity                          |
|:------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------:|:-----------:|:------------------------------------------------------------------------:|
| As of August 5, 2019, there were 46,662,179 shares of common stock, $0.01 par value, outstanding.                                                      | 4,66,62,179 | CARDINAL    | Shares Outstanding, Entity Common Stock                                  |
| The Company also derecognized existing deferred rent liabilities of $15,302.                                                                           | 15,302      | MONEY       | Rent Expense, Operating Leases                                           |
| The intangible assets acquired have a weighted average useful life of approximately nine years.                                                        | nine years  | DATE        | Intangible assets, Weighted Average  Useful Life, Acquired Finite lived  |
| The initial purchase price of $31,676 included $30,176 cash consideration paid upon acquisition,  funded primarily through borrowings under the Senior | 31,676      | MONEY       | Initial purchase price, Business  Combination, Consideration Transferred |
| Credit Facility, and a contingent earn out payment of up to $25,000 with an estimated fair  value of $1,500 as of the acquisition date.                | 30,176      | MONEY       | Cash consideration, Payments to  Acquire Businesses                      |



