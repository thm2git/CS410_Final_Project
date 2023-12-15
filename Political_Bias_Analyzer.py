## CS410 - Text Information Systems
## Course Project
## Arda Bedoyan
## Fall 20222

## Source: https://radimrehurek.com/gensim/index.html

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from collections import defaultdict
from gensim import corpora
from gensim import models
from gensim import similarities
from pathlib import Path

# Create a training set of articles and their main bias
# Bias selection was subjective
article_graph = {'Atlantic1': 'left',
               'Atlantic2': 'left',
               'Atlantic3': 'left',
               'Atlantic4': 'left',
               'Huff1': 'left',
               'Huff2': 'left',
               'Huff3': 'left',
               'Huff4': 'left',
               'MJ1': 'left',
               'MJ2': 'left',
               'MJ3': 'left',
               'MJ4': 'left',
               'MSNBC1': 'left',
               'MSNBC2': 'left',
               'MSNBC3': 'left',
               'MSNBC4': 'left',
               'Slate1': 'left',
               'Slate2': 'left',
               'Slate3': 'left',
               'Slate4': 'left',
               'Vox1': 'left',
               'Vox2': 'left',
               'Vox3': 'left',
               'Vox4': 'left',
               'CNN1': 'center',
               'CNN2': 'center',
               'CNN3': 'center',
               'CNN4': 'center',
               'NPR1': 'center',
               'NPR2': 'center',
               'NPR3': 'center',
               'NPR4': 'center',
               'BBC1': 'center',
               'BBC2': 'center',
               'BBC3': 'center',
               'BBC4': 'center',
               'CSM1': 'center',
               'CSM2': 'center',
               'CSM3': 'center',
               'CSM4': 'center',
               'Dispatch1': 'center',
               'Dispatch2': 'center',
               'Dispatch3': 'center',
               'Dispatch4': 'center',
               'Epoch1': 'center',
               'Epoch2': 'center',
               'Epoch3': 'center',
               'Epoch4': 'center',
               'Spectator1': 'right',
               'Spectator2': 'right',
               'Spectator3': 'right',
               'Spectator4': 'right',
               'Caller1': 'right',
               'Caller2': 'right',
               'Caller3': 'right',
               'Caller4': 'right',
               'Wire1': 'right',
               'Wire2': 'right',
               'Wire3': 'right',
               'Wire4': 'right',
               'Fox1': 'right',
               'Fox2': 'right',
               'Fox3': 'right',
               'Fox4': 'right',
               'Federalist1': 'right',
               'Federalist2': 'right',
               'Federalist3': 'right',
               'Federalist4': 'right',
               'Newsmax1': 'right',
               'Newsmax2': 'right',
               'Newsmax3': 'right',
               'Newsmax4': 'right'}

# Create a blank dictionary to store results in
bias_results = {}

# Initialize a starter variable to tell the program to run or stop
start = 'y'

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


#############end of function definition#######################


# Loop to run the program in
while (start == 'y' or start == 'Y' or start == 'yes' or start == 'Yes' or start == 'YES'):

    # user to input article to check
    # user must save text file of article script
    # and name it the same name as inputted into the program
    new_article = input('Enter the title of a article \n')
    keys = list(article_graph.keys())

    # check if the article already exists in our directory
    if (new_article in keys or new_article in list(bias_results.keys())):
        print('Article already exists in directory \n')
        if (new_article in list(bias_results.keys())):
            print('Biases: ' + str(bias_results[new_article]) + '\n')
        else:
            print('Bias: ' + str(article_graph[new_article]) + '\n')

        # allows user to rerun the program for existing articles
        # in case the addition of some articles affected its biases
        rerun = input('Do you want to run the article through the system again? \n')

        if (rerun == 'y' or rerun == 'Y' or rerun == 'yes' or rerun == 'Yes' or rerun == 'YES'):
            bias_analysis()
            start = input('Do you want to continue? \n')
        else:
            start = input('Do you want to continue? \n')

    # article does not exist in directory, run the program
    else:
        bias_analysis()
        start = input('Do you want to continue? \n')






