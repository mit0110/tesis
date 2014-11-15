from featmultinomial import FeatMultinomalNB
from feature_extraction import get_features


MAX_NGRAMS = 3

default_config = {
    # Corpus files
    'u_corpus_f': 'corpus/unlabeled_new_corpus.pickle',
    'test_corpus_f': 'corpus/test_new_corpus.pickle',
    'training_corpus_f': 'corpus/training_new_corpus.pickle',

    # Options to be displayed
    'number_of_classes': 30,
    'number_of_features': 30,

    # Classifier
    'classifier': FeatMultinomalNB(),

    # Features
    'feature_boost': 0.5,
    'em_adding_instances': 10,

    # Active learning instance selection function
    'get_next_instance': None,
    # Active learning feature selection functions
    'get_next_features': None,
    'handle_feature_prediction': None,
    # Active learning class selection function
    'get_class_options': None,
}
