import refo

from featureforge.feature import input_schema, output_schema
from featureforge.vectorizer import Vectorizer
from quepy import install

from literal_ner import LiteralNER


lner = LiteralNER()
freebase_app = install('quepyapp_freebase')


# @input_schema({'question': unicode})
# @output_schema(list(str), lambda l: len(l) > 0)
def postags(words):
    words, _ = words
    return [word.pos for word in words]


def lemmas(words):
    words, _ = words
    return [word.lemma for word in words]


def literal_ner(words):
    words, _ = words
    sentence = ' '.join(lemmas(words))
    ner_tuple = lner.find_ne(sentence)
    if ner_tuple:
        return [ner_tuple[0]] + ner_tuple[1]
    else:
        return ['No_NER']


def partial_matches(words):
    _, rules = words
    import ipdb; ipdb.set_trace()
    return rules


def get_features():
    return Vectorizer([postags, lemmas, partial_matches], sparse=True)