from featmultinomial import FeatMultinomalNB
from feature_extraction import get_features


MAX_NGRAMS = 3

default_config = {
    # Corpus files
    'u_corpus_f': 'corpus/unlabeled_corpus.pickle',
    'test_corpus_f': 'corpus/test_corpus.pickle',
    'training_corpus_f': 'corpus/training_corpus.pickle',

    # Options to be displayed
    'number_of_options': 30,

    # Classifier
    'classifier': FeatMultinomalNB,

    # Features
    'features': get_features(),
    'feature_boost': 0.5,
    'em_adding_instances': 10
}
