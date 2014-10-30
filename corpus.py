import pickle

class Corpus(object):
    def __init__(self):
        self.instances = None
        self.representations = None
        self.classes = None

    def load_from_file(self, filename):
        f = open(filename, 'r')
        self.instances, self.classes, self.representations = pickle.load(f)

    def save_to_file(self, filename):
        f = open(filename, 'r')
        pickle.dump((self.instances, self.classes, self.representations), f)