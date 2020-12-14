# CS 410 Final Project
jnhuss2, cnj3, edwardp3

### **Overview**

We are working on reproducing the paper, "Mining Causal Topics in Text Data: Iterative Topic Modeling with Time Series Feedback." Our code scrapes two datasets, IEM Price History data and the New York Times site map data. It uses these datasets to generate common topics in New York Times articles that are correlated with change in the IEM prices. The algorithm uses an iterative approach. It first uses LDA to generate topics. It then analyzes these topics to determine which topics and words are most correlated to change in the IEM data.  These results are used to generate priors, which are used as input to the next iteration of LDA.

### **How the Software is Implemented**

The primary functionality of the ITMTF algorithm is in the file *main.py*. In the following section, we break this file down into the various functions that are used to implement it. This file *data_reader.py* is used by *main.py* to read in the IEM and NYT data from the data files *IEMPrices.txt* and *articles.txt*.  Both of these txt documents are held in the data directory. *article_compiler.py* consists of the code that was used to scrape the NYT data and create *articles.txt*. *iem_csv_tranformer.py* consists of the code that was used to create *IEMPrices.txt* from the data held in *IEMPrices.csv*.

### **Functions**

In *main.py*: 
`read_data` - reads in data from articles.txt and IEMPrices.txt
`run_itmtf` - runs the ITMTF algorithm with nyt_data and runs methods to determine the significant topics
`get_topic_total_probability_by_date` - finds the probability of topics by date and creates a multidimensional array
`find_significant_topics` - finds the significant topics using the granger algorithm
`granger_significance_test` - runs grainger significance test off of a multidimensional array that is inputted as a parameter
`get_pos_neg_words` - states whether the word has a positive or negative correlation with the IEM prices
`determine_new_priors` - creates a weight for each word to be used next time LDA is run
`get_avg_confidence` - gets the average confidence amongst all of the words that have been chosen to be significant
`get_avg_purity` - gets the average purity amongst all of the topics

In *data_reader.py*
`get_iem_prices` - reads in IEMPrices.txt and returns a dictionary of date object keys and the iem normalized price values
`read_data_file`

### **Usage of Software / How to Run**
**APIs Used**

To program our software, we used Python. We used Python because of its simplicity and packages. For scraping the New York times article data, we used bs4's `BeautifulSoup`. We used the gensim library in multiple locations. In *main.py*, we used `gensim` to create a dictionary that acts as a vocabulary list for the entire corpus. The keys in the dictionary act as ids and the values are the strings of words, which is nice because each word gets an id associated with it. In *data_reader.py*, gensim is used to clean up the characters and words scraped from the New York times data. We used gensim methods, such as `strip_punctuation` and `remove_stopwords`. Next, we used `statsmodels.tsa.stattools` in *main.py* because it contains a method called `grangercausalitytests`, which we used for the granger test algorithm. We also used `scipy.stats` because it contains a method for the pearson correlation algorithm, `pearsonr`.

**How to Run**

To run the code, clone the repository and open it in your terminal. After this, install all of the needed packages and run `python main.py`.

```sh
$ pip install numpy
$ pip install datetime
$ pip install bs4
$ pip install gensim
$ pip install statsmodels
$ pip install scipy
$ python iem_csv_transformer.py
$ python main.py
```

Running this program took our computers about 25 minutes to complete. It runs the ITMTF algorithm a total of 4 times, each time varying the number of topics that is generated by LDA. Each time the ITMTF algorithm is run, its core functions are iterated 5 times (LDA runs 5 times).  With each iteration, the program prints out the topics it generated that are significant as well as the words that were most correlated to positive and negative movement in the IEM price.  Once ITMTF is run all 4 times, it prints out the average purity and causality for each iteration in each run of the algorithm.

### **Team Member Contributions**

**Jacob**
```sh
Scraped the IEM data. Wrote the code to read in IEM and New York Times data files. Wrote the backend code to find the significant words, differentiate if they are positive and negative, and make a list of words that "cause" or are related to changes in the IEM betting prices. 
```
**Chaitanya**
```sh
Worked with Edward to scrape the New York Times data. Cleaned the code, worked on implementing the method to find significant words, documented the code, and wrote the documentation
```
**Edward**
```sh
Worked with Chaitanya to scrape the New York Times data. (Ghosted us for the rest of the project)
```

### **Presentation**

You can view the final presentation using this YouTube link: https://youtu.be/5NiqwlT-tu4
It is also in the GitHub repository.

