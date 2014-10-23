import refo

from featureforge.feature import input_schema, output_schema
from featureforge.vectorizer import Vectorizer
from quepy import install

from literal_ner import LiteralNER


freebase_app = install('quepyapp_freebase')


# @input_schema({'question': unicode})
# @output_schema(list(str), lambda l: len(l) > 0)
def postags(words):
    words= words[0]
    return [word.pos for word in words]


def lemmas(words):
    words = words[0]
    return [word.lemma for word in words]


def partial_matches(words):
    rules = words[1]
    return rules


def literal_ners(words):
    if not words[2]:
        print "error", words
        return ('None',)
    return (words[2],)


def literal_ners_types(words):
    if not words[3]:
        print "error2", words
        return ('None',)
    return words[3]


def get_features():
    return Vectorizer(
        [postags, lemmas, partial_matches, literal_ners, literal_ners_types],
        sparse=True
    )