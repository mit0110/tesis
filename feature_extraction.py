from featureforge.feature import input_schema, output_schema
from featureforge.vectorizer import Vectorizer


# @input_schema({'question': unicode})
# @output_schema(list(str), lambda l: len(l) > 0)
def unigrams(question):
	return question.split()


def get_features():
	return Vectorizer([unigrams], sparse=False)