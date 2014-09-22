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
    return [word.pos for word in words]


def lemmas(words):
    return [word.lemma for word in words]


def literal_ner(words):
    sentence = ' '.join(lemmas(words))
    ner_tuple = lner.find_ne(sentence)
    if ner_tuple:
        return [ner_tuple[0]] + ner_tuple[1]
    else:
        return ['No_NER']


_EOL = None

def partial_matches(words):
    result = []
    for regex in freebase_app.partial_rules:
        match = refo.match(regex + refo.Literal(_EOL), words + [_EOL])
        if match:
            result.append(repr(regex))
    return result


def get_features():
    return Vectorizer([postags, lemmas, partial_matches], sparse=True)