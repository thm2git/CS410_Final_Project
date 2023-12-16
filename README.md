# CS410_Final_Project - Political Bias Analyzer
Teague Mitchell

# Overview:
The purpose of this project was to create a tool that could read the text of a political article and determine the bias of its source/writing. It takes in the input of an article and gives an output of a bias, either left, center, or right. It also gives the second most likely bias, which can be useful for identifying crossover between biases, such as center left or center right articles. This is accomplished by comparing similarities between the input article and the training dataset, and finding the highest likelihood. I think this kind of analysis can be useful for identifying similarites and differences in political outlooks in the US. I used abedoyans movie genre analyzer (linked below) as an inspiration and basis for starting this project, but found that many changes needed to be made to accomodate articles and bias being different from movies and genre.

# Getting Started:
The program uses the Gensim toolkit (https://radimrehurek.com/gensim/index.html)

Gensim runs on Linux, Windows and Mac OS X, and should run on any other platform that supports Python and NumPy. The documentation says that Gensim is tested with Python 3.6, 3.7, and 3.8. I used Python 3.8.7 on a Windows computer.

In your terminal, begin by running:
~~~
pip install --upgrade gensim
~~~

# The Training and Test Sets:
To analyze bias, a training set was first needed for the algorithm to base its conclusions off of. To accomplish this, I used the AllSides media bias chart and picked articles from major news sources in each section bias. I took 24 from left publications (classifying them as left), 8 from left leaning publications, 8 from center publications, 8 from right leaning publications (all 24 of which were classified as center biased), and 24 from right publications (which were classified as right biased). I also took half as many from each section of bias for the testing set (12, 4, 4, 4, 12). I used a much larger training and test set than abedoyan in part to accomodate for the fact that articles are shorter than movie scripts, so I felt more articles should help flesh out the training and testing.

The text file article_links contains links to all the articles used in the final version of this project. These articles text are also in text files marking their publication of origin for the training set, or their intended bias for the test set (as in for the 12 left articles in the test set they are named Left1 through Left 12). This is to make testing as easy as possible. There is also a text file for stop words so they can be filtered out.


# The Code
The code for the program is in the Python file, Political_Bias_Analyzer.py.

The code imports the necessary libraries below.
~~~
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from collections import defaultdict
from gensim import corpora
from gensim import models
from gensim import similarities
from pathlib import Path
~~~

The bias_analysis() function is where the analysis truly takes place. It reads in all of the article from the training set, removes common words, and tokenizes the documents. It then removes words that only occur once. A corpus is created that consists of all of the pre-processed scripts.

The Vector Space Model algorithm used in this code is Latent Semantic Indexing (LSI), sometimes referred to as Latent Semantic Analysis (LSA). The LSI code implemented was based off of the code from the Gensim documentation (https://tinyurl.com/2vrvvejf). LSI transforms documents from either a bag-of-words or Tf-Idf-weighted space, into a latent space of a lower dimensionality. According to Gensim documentation, dimensionality of 200–500 is recommended as the gold standard. Abedoyan chose to use 300, and after testing I confirmed it worked equally well as 200, 400, and 500, so I stuck with 300.

~~~
# Function to run the bias analysis using similarity queries #
# Based off of similarity query documentation from Gensim
# https://radimrehurek.com/gensim/auto_examples/core/run_similarity_queries.html#sphx-glr-auto-examples-core-run-similarity-queries-py
def bias_analysis():
    documents = []
    key_count = 0

    # read in the article scripts
    # the names of the txt files of the scripts should match the names in article_graph
    while (key_count < len(keys)):
        article = Path(keys[key_count] + '.txt').read_text(encoding="cp437")
        documents.append(article)
        key_count += 1

    # remove common words and tokenize
    stoplist = set(line.strip() for line in open('stop_words.txt'))

    texts = [
        [word for word in document.lower().split() if word not in stoplist]
        for document in documents
    ]

    # remove words that appear only once
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1

    texts = [
        [token for token in text if frequency[token] > 1]
        for text in texts
    ]

    # create a corpus of all of the documents
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # create an LSI model
    # LSI = Latent Semantic Indexing
    # Transforms documents from either bag-of-words or TfIdf-weighted space
    # into a latent space of a lower dimensionality.
    lsi = models.LsiModel(corpus, id2word=dictionary, num_topics=300)

    # read in the article script to run bias analysis on
    doc = Path(new_article + '.txt').read_text(encoding="cp437")
    doc = [word for word in doc.lower().split() if word not in stoplist]

    # convert the query to LSI space
    vec_bow = dictionary.doc2bow(doc)
    vec_lsi = lsi[vec_bow]  
    #print(vec_lsi)

    # transform corpus to LSI space and index it
    index = similarities.MatrixSimilarity(lsi[corpus])

    # perform a similarity query against the corpus
    sims = index[vec_lsi]

    # sort the results
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    # get the top two articles based on similarity
    top_choice = sims[0]
    second_choice = sims[1]

    # get the top two biases based on top three articles
    for doc_position, doc_score in sims:
        print(doc_score, "Article:", list(article_graph.keys())[doc_position])
        if doc_position == top_choice[0]:
            bias1 = list(article_graph.values())[doc_position]
        elif doc_position == second_choice[0]:
            bias2 = list(article_graph.values())[doc_position]


    # print out the results and remove duplicate biases
    print("\nBiases: ", end="")

    count = 1
    while(count < 3):
        if (count == 1):
            print(bias1 + " ", end="")
        if (count == 2 and bias2 != bias1):
            print(bias2 + " ", end="")
        count +=1

    print()

    temp_list = []
    biases = [bias1, bias2]

    for bias in biases:
        if bias not in temp_list:
            temp_list.append(bias)
~~~

After the model is run, the results are sorted and the biases corresponding to the top two articles most similar to the inputted article are gathered. After de-duplicating any repeated genres, the program prints out the results of the bias analysis.

# Evaluation:
I found working on this project really interesting and taught me a lot about topic modeling. In hindsight I would have liked to try more solutions and see how they compare to this method. In particular I tried building a web scraper to make the training set much larger but found that it would take me too long for the scope of this project. Just manually finding articles for the training set took me a very long time due to me wanting a fairly large set. In addition, I initially collected articles over time as I was working on the project but then realized my articles were heavily biased by the time I collected them (as in, the same news is covered in a particular week so if all of my right leaning articles are from a different week than the rest than that will bias all articles in the test set from that period to match the right leaning articles). So, late into the project I scrapped my entire training and test set and recollected articles in one day all from the last week or so. That way bias could be found in topics and language in different publications rather than just time passing causing new news to be relevant.

I also think the similarity measure used for this project probably isn't ideal for political articles due to their relative briefness. Since they have a lot of breadth with less depth I think a topic modeling that considered more articles at once for comparison would likely perform better, so if I had more time I would have liked to explore that option more. I especially would have liked to experiment with LDA to see if that produced fiferences.

I did find it fun to use another person's previous project as a basis to see how it could work under a different angle. It was tough because with updates and other potential qualifiers I found it initially quite difficult to get their code functioning, and I had to do some tough debugging. In particular, the codec decoding with text files was a big challenge that I eventually fixed by forcing different codecs until I found one that worked well and seemed to match. Overall though, I feel I wasn't able to innovate as much as I would have liked due to these difficulties, so I found that to be the metaphorical double-edged sword aspect of it.

Overall though, I was impressed by the accuracy of the analyzer. I think this could be due to a lot of reasons, not all of them positive. For one I feel the test set may have been a bit too similar to the training set and thus the versatility was not tested for. I find it interesting that the leaning right articles struggled to classify any as having a right bias, and that left biases were a lot more common among publications called center biased. It's also interesting to me that many articles have a secondary label that is more than one step away, such as left and right. I think this shows how often left and right perspectives can be more similar than a center approach. In general, I think this project is a good example of it being very possible for algorithms to assist us in understanding political bias.

![image](https://github.com/thm2git/CS410_Final_Project/assets/111996907/e50b96f8-ac03-4bd7-a48b-d63df7396ff7)
![image](https://github.com/thm2git/CS410_Final_Project/assets/111996907/970fa68c-8b05-46c3-a56c-2207fc4f2ccb)
![image](https://github.com/thm2git/CS410_Final_Project/assets/111996907/96ddd482-7931-45e3-91da-1ab24ac3986c)


# Team Contributions:
I did this alone

# Sources:
https://radimrehurek.com/gensim/auto_examples/core/run_topics_and_transformations.html#sphx-glr-auto-examples-core-run-topics-and-transformations-py

https://radimrehurek.com/gensim/auto_examples/core/run_similarity_queries.html#sphx-glr-auto-examples-core-run-similarity-queries-py

https://en.wikipedia.org/wiki/Latent_semantic_analysis#Latent_semantic_indexing

https://github.com/abedoyan/CS410_CourseProject

https://www.allsides.com/media-bias/media-bias-chart
