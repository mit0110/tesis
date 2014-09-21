from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

MAX_NGRAMS = 3

default_config = {
    # Corpus files
    'u_corpus_f': 'corpus/unlabeled_corpus.pickle',
    'test_corpus_f': 'corpus/test_corpus.pickle',
    'training_corpus_f': 'corpus/training_corpus.pickle',

    # Options to be displayed
    'number_of_options': 30,

    # Classifier
    'classifier': MultinomialNB,

    # Features
    'features': CountVectorizer(ngram_range=(1, MAX_NGRAMS)),
    # from feature_extraction import get_features
    # FEATURES = FeatureUnion([('custom', get_features()),
    #                          ('n_grams', countv)])),

}