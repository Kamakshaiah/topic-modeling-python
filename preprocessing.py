def ReadCSV(path, var):
    ''' Importing CSV files '''
    
    import pandas as pd
    data = pd.read_csv(path)
    data_text = data[[var]]
    data_text['index'] = data_text.index
    documents = data_text
    return documents

def lemmatize_stemming(text):
    ''' Does only lemmatization '''
    
    from gensim.parsing.preprocessing import STOPWORDS
    ps = PorterStemmer()
    return ps.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def Preprocess(text):
    ''' Prepprocessing text - does both lemmatising and stemming '''
    
    import gensim
    from gensim.utils import simple_preprocess
    from gensim.parsing.preprocessing import STOPWORDS
    
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

def PreprocessDocs(text, var):
    ''' Does preprocessing with lemmatization, stemming and tokennization. '''
    import gensim
    from gensim.utils import simple_preprocess
    from gensim.parsing.preprocessing import STOPWORDS
    from nltk.stem import WordNetLemmatizer, SnowballStemmer, PorterStemmer
    import numpy as np
    np.random.seed(2018)
    import nltk

    def lemmatize_stemming(text):
        ps = PorterStemmer()
        return ps.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


    def preprocess(text):
        
        result = []
        for token in gensim.utils.simple_preprocess(text):
            if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
                result.append(lemmatize_stemming(token))
                return result

    processed_docs = text[var].map(preprocess)

    return processed_docs[:10]
    

if __name__ == '__main__':
##    path = 
##    path = os.getcwd()
##    
##    ReadCSV(path)
