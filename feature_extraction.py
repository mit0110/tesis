import refo

from featureforge.feature import input_schema, output_schema
from featureforge.vectorizer import Vectorizer
from quepy import install


freebase_app = install('quepyapp_freebase')


# @input_schema({'question': unicode})
# @output_schema(list(str), lambda l: len(l) > 0)
def postags(words):
    words = words[0]
    return set(['POS:' + word.pos for word in words])


def lemmas(words):
    words = words[0]
    return set(['Lemma:' + word.lemma for word in words])


def bigrams(words):
    words = words[0]
    return ['Bigram:{0},{1}'.format(words[i].lemma, words[i+1].lemma)
            for i in range(len(words)-1)]


def mixed_bigrams(words):
    words = words[0]
    lemma_first = ['MBigram:{0},{1}'.format(words[i].lemma, words[i+1].pos)
                    for i in range(len(words)-1)]
    pos_first = ['MBigram:{0},{1}'.format(words[i].pos, words[i+1].lemma)
                 for i in range(len(words)-1)]
    return lemma_first + pos_first


def trigrams(words):
    words = words[0]
    return ['Trigram:{0},{1},{2}'.format(words[i].lemma, words[i+1].lemma,
                                         words[i+2].lemma)
            for i in range(len(words)-2)]


def partial_matches(words):
    rules = words[1]
    return ['Rule:' + str(r) for r in rules]


def literal_ners(words):
    if not words[2]:
        print "error", words
        return ('Named Entity:None',)
    return ('Named Entity:' + words[2],)


def literal_ners_types(words):
    if not words[3]:
        print "error2", words
        return ('Named Entity Type:None',)
    return ['Named Entity Type:' + t for t in words[3]]


def get_features():
    return Vectorizer(
        [postags, lemmas, partial_matches, literal_ners, literal_ners_types,
         bigrams, trigrams, mixed_bigrams],
        sparse=True
    )
