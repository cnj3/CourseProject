# CS 410 Final Project
jnhuss2, cnj3, edwardp3

### **Overview**

We are working on reproducing the paper, "Mining Causal Topics in Text Data: Iterative Topic Modeling with Time Series Feedback." Our code scrapes two datasets, IEM Price History data and New York Times site map data. It uses these datasets to find a correlation between the IEM Price history and New York Times to find the correlation between words and prices by date. Next, we use this information to find topics / words in articles on New York Times that "cause" or are related to changes in the IEM betting prices

Our program could be used to find correlations between text and betting price histories. It could be used to see what topics throughout any text item affect prices of a good, service, stock, or bet.

### **How The Software Is Implemented**

The code for the software is divided into different sections of *main.py* and the different files in the repository, including *article_compiler.py*. The code to read the New York times data and parse it is in *article_compiler.py*. The code to read the IEM data and parse it is in *iem_csv_tranformer.py*. These two files write to two text files in the data folder, *articles.txt", for the New York times article data, and *IEMPrices.txt*, for the IEM price data. The file, *data_reader.py*, has methods to obtain information from these two text files, like getting a dictionary of IEM prices. 

### ADD MORE HERE

### **Usage of Software / How to Run**

To program our software, we used Python. We used Python because of its simplicity and packages. For scraping the New York times article data, we used bs4's `BeautifulSoup`. We used the gensim library in multiple locations. In *main.py*, we used `gensim` to create a dictionary that acts as a vocabulary list for the entire corpus. The keys in the dictionary act as ids and the values are the strings of words, which is nice because each word gets an id associated with it. In *data_reader.py*, gensim is used to clean up the characters and words scraped from the New York times data. We used gensim methods, such as `strip_punctuation` and `remove_stopwords`. Next, we used `statsmodels.tsa.stattools` in *main.py* because it contains a method called `grangercausalitytests`, which we used for the granger test algorithm. We also used `scipy.stats` because it contains a method for the pearson correlation algorithm, `pearsonr`.

To run the code, clone the repository and open it in your terminal. Then, run `python iem_csv_transformer.py` and `python article_compiler.py` to scrape the IEM data and New York Times data. This only needs to be done the first time. After this, run `python main.py`.
```sh
$ pip install numpy
$ pip install datetime
$ pip install bs4
$ pip install gensim
$ pip install statsmodels
$ pip install scipy
$ python main.py
```

### **Team Member Contributions**

**Jacob**
```sh
Scraped the IEM data. Wrote the backend code to find the significant words, differentiate if they are positive and negative, and make a list of words that "cause" or are related to changes in the IEM betting prices. 
```
**Chaitanya**
```sh
Worked with Edward to scrape the New York Times data. Cleaned the code, worked on implementing the method to find significant words, documented the code, and wrote the documentation
```
**Edward**
```sh
Worked with Chaitanya to scrape the New York Times data.
```