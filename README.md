# EDGAR10-Q Dataset

This dataset is built from 10-Q documents (Quarterly Reports) of publicly listed companies on the SEC. To access these documents, follow [this link](https://www.sec.gov/os/accessing-edgar-data). The script ie_parser.py contains code to create key-value pairs from “facts” in the XBRL files by mapping the attribute’s value with its corresponding label.


Some Statistics about the corpus are given below : 

Table 1 : Basic Statistics of the dataset

| Statistics about the corpus             | Numbers |
|-----------------------------------------|---------|
| Total  # of Documents in Corpus         | 19468   |
| Documents in which entities were tagged | 18752   |
| Number of Paragraphs                    | 670767  |
| Total Words in paragraphs               | 19468   |
| Number of Sentences                     | 973655  |
| Total Entities                          | 1783617 |
| Total Labels                            | 3528849 |


Table 2 : Sentence Wise Statistics of the dataset


| Sentences with # of Entities | # of Sentences | Average Word Length of the Sentence |
|------------------------------|---------------:|-------------------------------------|
| 1 Entity                     | 477341         | 33.50                               |
| 2 Entity                     | 336552         | 38.94                               |
| 3 Entity                     | 64449          | 51.05                               |
| 4 Entity                     | 68144          | 54.87                               |
| 5 Entity                     | 12683          | 58.96                               |
| 6 and above                  | 14486          | 78.71                               |


Table 3 : Statistics of the entities in the dataset

|    Entity Types   |  Counts |
|:-----------------:|:-------:|
| monetary ItemType | 1214552 |
| percent ItemType  | 216113  |
| pure ItemType     | 14017   |
| duration ItemType | 84071   |
| shares ItemType   | 180778  |
| integer ItemType  | 74086   |


Table 4 :  Instance of the Dataset.


|                                                                        Paragraph                                                                       |    value    | entity type |                          Labels for each entity                          |
|:------------------------------------------------------------------------------------------------------------------------------------------------------:|:-----------:|:-----------:|:------------------------------------------------------------------------:|
| As of August 5, 2019, there were 46,662,179 shares of common stock, $0.01 par value, outstanding.                                                      | 4,66,62,179 | CARDINAL    | Shares Outstanding, Entity Common Stock                                  |
| The Company also derecognized existing deferred rent liabilities of $15,302.                                                                           | 15,302      | MONEY       | Rent Expense, Operating Leases                                           |
| The intangible assets acquired have a weighted average useful life of approximately nine years.                                                        | nine years  | DATE        | Intangible assets, Weighted Average  Useful Life, Acquired Finite lived  |
| The initial purchase price of $31,676 included $30,176 cash consideration paid upon acquisition,  funded primarily through borrowings under the Senior | 31,676      | MONEY       | Initial purchase price, Business  Combination, Consideration Transferred |
| Credit Facility, and a contingent earn out payment of up to $25,000 with an estimated fair  value of $1,500 as of the acquisition date.                | 30,176      | MONEY       | Cash consideration, Payments to  Acquire Businesses                      |



