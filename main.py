import re
import os
import math

def tokenization(text): # Tokenizes given text
    text = re.sub("n\'t", ' not ', text.lower())  # converts n't to not
    text = re.sub("[\'\"]s", ' ', text.lower())  # removes 's or "s
    text = re.sub("[\`\'\"\?\!\@\(\)\.\,\\\/]", ' ', text.lower())  # '"/\@!?"' characters are deleted
    tokens = re.split('\s+', text) # splits the text around spaces.
    stop_words = ['ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than']
    tokens = [token for token in tokens if token not in stop_words] # stop words are not considered as tokens with the idea that they do not contribute to sentiment analysis.
    return tokens


def readFiles(trainFilesP, trainFilesN): #trainFilesP are positive file names, trainFilesN are negative file names.
    pwholeText = '' # All positive text will be cumulated with this variable
    nwholeText = '' # All negative text will be cumulated with this variable
    pdocuments = [] # Positive Tokens are tokenized and collected seperately
    ndocuments = [] # Negative tokens are tokenized and collected seperately
    for index, posFile in enumerate(trainFilesP): # read each positive comment
        with open(f'./data/train/pos/{posFile}') as f:
            text = f.read() # read it
            tokens = tokenization(text) # tokenize the text
            pdocuments.append(tokens) # add tokens as a list to pdocuments
            pwholeText += (' ' + text)  # union all positive texts


    for index, negFile in enumerate(trainFilesN):
        with open(f'./data/train/neg/{negFile}') as f:
            #             print(index, negFile)
            text = f.read() # read it
            tokens = tokenization(text)# tokenize the text
            ndocuments.append(tokens)# add tokens as a list to ndocuments
            nwholeText += (' ' + text) # union all negative texts

    return pdocuments, ndocuments, pwholeText, nwholeText


def createVocabulary(positiveDocuments, negativeDocuments): # given a positive and a negative text, tokenize each and form a vocabulary
    vocabulary = set()
    positiveTokens = []
    negativeTokens = []

    for document in (positiveDocuments): # iterate over positive documents
        vocabulary = vocabulary.union(document) # add tokens to vocabulary as new documents are read
        positiveTokens = positiveTokens + document

    for document in (negativeDocuments): # iterate over negative documents
        vocabulary = vocabulary.union(document)
        negativeTokens = negativeTokens + document

    return positiveTokens, negativeTokens, vocabulary  # return positive tokens, negative tokens and all tokens

def bernoulli(word, document):
    # Given a word, I find the probability of that word consisting in documents.
    # each element of document is a list of tokens.
    counter = 0 # counter counts the total number of documents the word passes in
    for comment in document:
        if (word in comment):
            counter += 1
    return (1 + counter) / (2 + len(document)) # add1-smoothing is applied.

def guess_bernoulli(document, vocabulary):
    # Given a document, it forms the tokens from document
    # For each word in vocabulary, if word is contained in the document, we add the probability of the word contained in positive documents
    # If not, then I add (1- probability)
    tokens = set(tokenization(document))
    negative_probability = 0
    positive_probability = 0
    for word in vocabulary: # iterate over vocabulary
        probP = bernoulli_probs_P[word] # for each word calculate probability of occurrence in positive documents
        if (word in tokens): # If the word is in tokens, add probability else ( 1- probability )
            positive_probability += math.log(probP) # Logarithm is applied to overcome underflow
        else:
            positive_probability += math.log(1 - probP)

        probN = bernoulli_probs_N[word]
        if (word in tokens):
            negative_probability += math.log(probN)
        else:
            negative_probability += math.log(1 - probN)

    numP = len(trainFilesP) / len(trainFilesP + trainFilesN)
    numN = len(trainFilesN) / len(trainFilesP + trainFilesN)
    positive_probability += math.log(numP)
    negative_probability += math.log(numN)

    if (positive_probability >= negative_probability):
        return 1
    else:
        return 0


def test_bernoulli(testSet, vocabulary): # Given a test set and our vocabulary. Algorithm predicts the class of each set.

    result_table = [[0, 0], [0, 0]] # Result table is 2by2 matrix.
    # First row is estimate 0 and second  row is estimate 1
    # First column is true value 0 (negative) and second column is value 1 (positive)
    for test in testSet.items():
        path = f'./data/test/pos/{test[0]}' if test[1] == 1 else f'./data/test/neg/{test[0]}'
        with open(path, 'r') as f:
            text = f.read()
            estimate = guess_bernoulli(text, vocabulary) # estimate is 1 if algorithm thinks it is positive
            result_table[estimate][test[1]] += 1
    return result_table


def multinomial_nb(text, numberoftokensclassj, counter, V):
    tokens = tokenization(text) # tokenize text
    totalSum = 0

    for word in tokens:
        numberoftimes = counter[word]
        probability = (numberoftimes + 1) / (numberoftokensclassj + V)
        totalSum += math.log(probability)
        # Logarithm is used to prevent any underflow that may result in
        # floating point multiplication
    return totalSum


def test_multinomial(testSet, V, positiveCounter, negativeCounter):

    result_table = [[0, 0], [0, 0]]
    for test in testSet.items():
        path = f'./data/test/pos/{test[0]}' if test[1] == 1 else f'./data/test/neg/{test[0]}'
        with open(path) as f:
            testText = f.read()
            probP = multinomial_nb(testText, len(posTokens), positiveCounter, V) # Calculate probability in negative texts
            probN = multinomial_nb(testText, len(negTokens), negativeCounter, V) # Calculate probability in positive texts
            if (probP > probN):
                estimate = 1
            else:
                estimate = 0
            result_table[estimate][test[1]] += 1
    return result_table


def binary_nb(text, numberoftokensclassj, counter, V):
    tokens = tokenization(text)
    totalSum = 0
    c = 0

    for word in set(tokens): # To differentiate from multinomial NB, this algorithm iterates over set of tokens. Which eliminates effect of more occurence of a word in a document
        numberoftimes = counter[word]
        probability = (numberoftimes + 1) / (numberoftokensclassj + V)
        totalSum += math.log(probability)  # Logarithm is used to prevent
        # any underflow that may result in floating point multiplication

    return totalSum


def test_binary(testSet, V, positiveCounter, negativeCounter):
    # tp = tn = fp = fn = 0
    result_table = [[0, 0], [0, 0]]
    for test in testSet.items():
        path = f'./data/test/pos/{test[0]}' if test[1] == 1 else f'./data/test/neg/{test[0]}'
        with open(path) as f:
            testText = f.read()
            probP = binary_nb(testText, len(posTokens), positiveCounter, V) # Calculate probability in positive texts
            probN = binary_nb(testText, len(negTokens), negativeCounter, V) # Calculate probability in negative texts
            if (probP > probN):  # If the positive probability is bigger than negative
                estimate = 1  # make estimate 1 meaning positive
            else:
                estimate = 0
            result_table[estimate][test[1]] += 1
    return result_table


def precision(tp, fp):
    return (tp / (tp + fp))


def recall(tp, fn):
    return (tp / (tp + fn))


def f1Measure(precision, recall):
    return 2 * precision * recall / (precision + recall)


def classPositive_statistics(result_table):
    # Given a results table
    # create statistics for class positive
    # First row is estimate negative, second row is estimate positive
    # First column is truth negative, second column is truth positive
    tp = result_table[1][1]
    tn = result_table[0][0]
    fp = result_table[1][0]
    fn = result_table[0][1]
    p_precision = precision(tp, fp)
    p_recall = recall(tp, fn)
    p_fmeasure = f1Measure(p_precision, p_recall)
    return tp, tn, fp, fn,  p_precision, p_recall, p_fmeasure

def classNegative_statistics(result_table):
    # Given a results table
    # create statistics for class negative
    # First row is estimate negative, second row is estimate positive
    # First column is truth negative, second column is truth positive
    tp = result_table[0][0]
    tn = result_table[1][1]
    fp = result_table[0][1]
    fn = result_table[1][0]
    n_precision = precision(tp, fp)
    n_recall = recall(tp, fn)
    n_fmeasure = f1Measure(n_precision, n_recall)
    return tp, tn, fp, fn, n_precision, n_recall, n_fmeasure


trainFilesP = os.listdir('./data/train/pos') # First read positive training data
trainFilesN = os.listdir('./data/train/neg') # Secondly read negative training data
if '.DS_Store' in trainFilesP:
    trainFilesP.remove('.DS_Store') # These files caused some problems, since I used os.listdir function of the python
if '.DS_Store' in trainFilesN:
    trainFilesN.remove('.DS_Store')

testPos = os.listdir('./data/test/pos')
testNeg = os.listdir('./data/test/neg')
if '.DS_Store' in testPos:
    testPos.remove('.DS_Store')  # remove DS_Store files
if '.DS_Store' in testNeg:
    testNeg.remove('.DS_Store')
testSet = dict(
    list({testFile: 1 for testFile in testPos}.items()) + list({testFile: 0 for testFile in testNeg}.items()))

positiveDocuments, negativeDocuments, positiveText, negativeText = readFiles(trainFilesP, trainFilesN) # read data files, tokenize all and form one text out of them

posTokens, negTokens, vocabulary = createVocabulary(positiveDocuments, negativeDocuments) #
V = len(vocabulary)
bernoulli_probs_P = {}
bernoulli_probs_N = {}
# For each vocabulary word
# Calculate document frequencies of words for both classes
# bernoulli_probs_N[word] = number of positive documents word is contained / all positive documents
# with addition of laplace smoothing
for word in vocabulary:
    probP = bernoulli(word, positiveDocuments)
    probN = bernoulli(word, negativeDocuments)
    bernoulli_probs_P[word] = probP
    bernoulli_probs_N[word] = probN


################################## BERNOULLI NB #########################################################
result_bernoulli = test_bernoulli(testSet, vocabulary)
pos_tp, pos_tn,pos_fp, pos_fn, pos_precision, pos_recall, pos_f1measure = classPositive_statistics(result_bernoulli)
neg_tp, neg_tn, neg_fp, neg_fn, neg_precision, neg_recall, neg_f1measure = classNegative_statistics(result_bernoulli)

micro_precision = precision(pos_tp+neg_tp, pos_fp+neg_fp)
macro_precision = (pos_precision + neg_precision) / 2
micro_recall = recall(pos_tp+neg_tp, pos_fn+neg_fn)
macro_recall = (pos_recall + neg_recall) / 2
micro_f1 = f1Measure(micro_precision, micro_recall)
macro_f1 = (pos_f1measure + neg_f1measure) / 2
print('Bernoulli')
print('Micro precision, recall, f1: ', micro_precision, micro_recall, micro_f1)
print('Macro precision, recall, f1: ', macro_precision, macro_recall, macro_f1)



#################################### MULTINOMIAL NB #######################################################
# Counter of tokens, used for multinomial and binary NB
from collections import Counter
positiveCounter = Counter(posTokens) # count number of occurrence of each positive token
negativeCounter = Counter(negTokens)# count number of occurrence of each negative token

result_multinomial = test_multinomial(testSet, len(vocabulary), positiveCounter, negativeCounter)
pos_tp, pos_tn,pos_fp, pos_fn, pos_precision, pos_recall, pos_f1measure = classPositive_statistics(result_multinomial)
neg_tp, neg_tn, neg_fp, neg_fn, neg_precision, neg_recall, neg_f1measure = classNegative_statistics(result_multinomial)

micro_precision = precision(pos_tp+neg_tp, pos_fp+neg_fp)
macro_precision = (pos_precision + neg_precision) / 2
micro_recall = recall(pos_tp+neg_tp, pos_fn+neg_fn)
macro_recall = (pos_recall + neg_recall) / 2
micro_f1 = f1Measure(micro_precision, micro_recall)
macro_f1 = (pos_f1measure + neg_f1measure) / 2
print('Multinomial')
print('Micro precision, recall, f1: ', micro_precision, micro_recall, micro_f1)
print('Macro precision, recall, f1: ', macro_precision, macro_recall, macro_f1)

##################################### BINARY NB #######################################################
result_binary = test_binary(testSet, len(vocabulary), positiveCounter, negativeCounter)
pos_tp, pos_tn,pos_fp, pos_fn, pos_precision, pos_recall, pos_f1measure = classPositive_statistics(result_binary)
neg_tp, neg_tn, neg_fp, neg_fn, neg_precision, neg_recall, neg_f1measure = classNegative_statistics(result_binary)

micro_precision = precision(pos_tp+neg_tp, pos_fp+neg_fp) # Micro averaged precision add numbers and calculate precision
macro_precision = (pos_precision + neg_precision) / 2 # Macro averaged precision, mean of precision of both classes.
micro_recall = recall(pos_tp+neg_tp, pos_fn+neg_fn) # Micro averaged recall, add numbers and calculate recall
macro_recall = (pos_recall + neg_recall) / 2 # Macro averaged recall , mean of recall of both classes.
micro_f1 = f1Measure(micro_precision, micro_recall)  # Micro averaged f1, add numbers and calculate f1
macro_f1 = (pos_f1measure + neg_f1measure) / 2 # Macro averaged f1 , mean of f1 of both classes.
print('Binary')
print('Micro precision, recall, f1: ', micro_precision, micro_recall, micro_f1)
print('Macro precision, recall, f1: ', macro_precision, macro_recall, macro_f1)