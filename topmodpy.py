# package for topic modeling
# to dos: need to implement dict for 'printtopics' method
#           need to implement dict in 'printtopicswithweights'


def help():
    ''' shows help for topmodpy module '''

    print("FileImport - arg: 'path'; return: dataset")
    print('GetHead - First few records of data set (imported using method - FileImport()')    
    print('CreateVariable - args: data, var; return data variable [arg: var] (of interest)')
    print('CleanVar - args: var; removes special characters and convert data (words in docs) into lower case letters')
    print('CreateWordcloudImg - args: var; returns wordcloud image')
    print('CreateCountVector - args: var; computes counts for each word for given input variable [arg: var]')
    print('PrintTopics - args: var, nw, nt; prints topics for inputs nt (number of words) and nt (number of topics) for a given input variable (var)')
    print('PrintTopicsWithWeights - args: var, nt, nw; prints (return not implemented) for nt (number of words) and nt (number of topics) for a given input variable (var)')
    print("LDA - args: count_data; returns (LDA) model for input data variable (count_data). Use CreateCountVector method to compute 'count_data'.")
    print("CountVectorizer - A helpler function for 'MakeHTML'")
    print("MakeHTML - args: var; creates interactive HTML doc with added visuals for each topic.")
    print("OpenHTMLFile - no args; a helper function to open HTML document created by method 'MakeHTML'. ")
        
    
def FileImport(path):
    ''' arg: 'path'; return: dataset '''

    # imports data file returns 'data' object

    import pandas as pd
    data = pd.read_excel(path)
    return data

def GetHead(data):

    ''' GetHead - First few records of data set (imported using method - FileImport() '''

    # gives the head [first few records of data file] information
    
    print(data.head())
    

def CreateVariable(data, var):

    ''' CreateVariable - args: data, var; return data variable [arg: var] (of interest) '''
    
    import pandas as pd

    # importing data
    
    datavar = data[var]
##    print(f'Variable {var} created! Following is the few records of variable' )
##    print(datavar.head())
    return datavar

def CleanVar(var):

    ''' CleanVar - args: var; removes special characters and convert data (words in docs) into lower case letters '''

    # clearning
    import re
    cleanedvar = var.map(lambda x: re.sub('[,\.!?]', '', x))
    cleanedvar = cleanedvar.map(lambda x: x.lower())
##    print("Cleaning done! Following are the few records of cleaned variable")
##    print(cleanedvar.head())
    return cleanedvar

def CreateWordcloudImg(var):

    ''' CreateWordcloudImg - args: var; returns wordcloud image '''

    # word cloud

    from wordcloud import WordCloud

    longstring = ','.join(list(var.values))
    wordcloud = WordCloud(background_color="white", max_words=5000, contour_width=3, contour_color='steelblue')
    wcplot = wordcloud.generate(longstring)

    ##wordcloud.to_image()

    import matplotlib.pyplot as plt
    plt.imshow(wcplot, interpolation='bilinear')
    plt.show()

def CreateCountVector(var):

    ''' CreateCountVector - args: var; computes counts for each word for given input variable [arg: var] '''

    # performs LDA

    from sklearn.feature_extraction.text import CountVectorizer

    count_vectorizer = CountVectorizer(stop_words='english')
    count_data = count_vectorizer.fit_transform(var)
    return count_data

def PrintTopics(var, nt, nw):

    ''' PrintTopics - args: var, nw, nt; prints topics for inputs nt (number of words) and nt (number of topics) for a given input variable (var) '''

    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np

    import seaborn as sns
    sns.set_style('whitegrid')

    count_vectorizer = CountVectorizer(stop_words='english')
    count_data = count_vectorizer.fit_transform(var)

    ##plot_10_most_common_words(count_data, count_vectorizer)

    import warnings
    warnings.simplefilter("ignore", DeprecationWarning)

    # Load the LDA model from sk-learn
    from sklearn.decomposition import LatentDirichletAllocation as LDA

    def print_topics(model, count_vectorizer, n_top_words):
        words = count_vectorizer.get_feature_names()
        for topic_idx, topic in enumerate(model.components_):
            print("\nTopic #%d:" % topic_idx)
            print(" ".join([words[i]for i in topic.argsort()[:-n_top_words - 1:-1]]))

    # Tweak the two parameters below
    number_topics = nt
    number_words = nw

    # Create and fit the LDA model
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(count_data)

    # Print the topics found by the LDA model
    print("Topics found via LDA:")
    topics = print_topics(lda, count_vectorizer, number_words)
    return topics

def PrintTopicsWithWeights(var, nt, nw):

    ''' PrintTopicsWithWeights - args: var, nt, nw; prints (return not implemented) for nt (number of words) and nt (number of topics) for a given input variable (var) '''

    finalvar = CleanVar(var)
    
    doc_complete = []
    [doc_complete.append(d) for d in finalvar]

    import nltk
    ##nltk.download()
    from nltk.corpus import stopwords
    from nltk.stem.wordnet import WordNetLemmatizer
    import string
    stop = set(stopwords.words('english'))
    exclude = set(string.punctuation)
    lemma = WordNetLemmatizer()

    def clean(doc):
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
        normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
        return normalized

    doc_clean = [clean(doc).split() for doc in doc_complete]

    import gensim
    from gensim import corpora

    # Creating the term dictionary of our courpus, where every unique term is assigned an index.
    dictionary = corpora.Dictionary(doc_clean)

    # Converting list of documents (corpus) into Document Term Matrix using dictionary prepared above.
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

    # Creating the object for LDA model using gensim library
    Lda = gensim.models.ldamodel.LdaModel

    # Running and Trainign LDA model on the document term matrix.
    ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)

    print(ldamodel.print_topics(num_topics=nt, num_words=nw))
    
    
def LDA(count_data):

    ''" LDA - args: count_data; returns (LDA) model for input data variable (count_data). Use CreateCountVector method to compute 'count_data'."''

    from sklearn.decomposition import LatentDirichletAllocation as LDA
    # Tweak the two parameters below
    number_topics = 5
    number_words = 10

    # Create and fit the LDA model
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(count_data)

    return lda

def CountVectorizer():

    ''" A helpler function for 'MakeHTML' "''
    
    from sklearn.feature_extraction.text import CountVectorizer
    import numpy as np

    import seaborn as sns
    sns.set_style('whitegrid')

    count_vectorizer = CountVectorizer(stop_words='english')
    return count_vectorizer

def MakeHTML(var):

    ''" MakeHTML - args: var; creates interactive HTML doc with added visuals for each topic. "''
    
    from pyLDAvis import sklearn as sklearn_lda
    import pickle
    import os
    import pyLDAvis

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation as LDA
    import numpy as np

    import seaborn as sns
    sns.set_style('whitegrid')

    count_vectorizer = CountVectorizer(stop_words='english')
    count_data = count_vectorizer.fit_transform(var)

    import warnings
    warnings.simplefilter("ignore", DeprecationWarning)

    # Tweak the two parameters below
    number_topics = 5
    number_words = 10

    # Create and fit the LDA model
    lda = LDA(n_components=number_topics, n_jobs=-1)
    lda.fit(count_data)
    
    LDAvis_data_filepath = os.path.join(os.getcwd() + '\\ldavis_prepared_'+ str(number_topics))


    LDAvis_prepared = sklearn_lda.prepare(lda, count_data, count_vectorizer)
    with open(LDAvis_data_filepath, 'wb') as f:
        pickle.dump(LDAvis_prepared, f)
        
    # load the pre-prepared pyLDAvis data from disk

    with open(LDAvis_data_filepath, 'rb') as f:
        LDAvis_prepared = pickle.load(f)

    pyLDAvis.save_html(LDAvis_prepared, LDAvis_data_filepath +'.html')

def OpenHTMLFile():

    ''" OpenHTMLFile - no args; a helper function to open HTML document created by method 'MakeHTML'. "''
    
    import webbrowser
    import os

    number_topics = 5
    LDAvis_data_filepath = os.path.join(os.getcwd() + '\\ldavis_prepared_'+ str(number_topics))

    url = LDAvis_data_filepath +'.html'
    new = 2

    webbrowser.open(url,new=new)


##if __name__ == '__main__':
##    data = FileImport("D:\\GSIB\MBA-Fintech\\assignments\\topicmodeling\\Hadoop.xlsx")
####    GetHead(data)
##    var = CreateVariable(data, "Abstract")
####    var.head()
##    cleanedvar = CleanVar(var)
####    clearnedvar.head()
####    CreateWordcloudImg(cleanedvar)
####    countdata = CreateCountVector(cleanedvar)
##    PrintTopics(cleanedvar)
####    
####    lda = LDA(countdata)
####    cv = CountVectorizer()
####
####    MakeHTML(var)
####    OpenHTMLFile()
##    PrintTopicsWithWeights(var, 5, 6)

