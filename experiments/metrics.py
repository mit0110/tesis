import pickle

class Metric(object):
    """Represents information and a method to obtained from a session file.

    To create a custom metric, redefine the get_from_session method saves in
    self.info the content of the metric extracted from self.session.
    """
    def __init__(self):
        self.name = self.__class__.__name__.lower()

    def get_from_session(self):
        raise NotImplementedError

    def get_from_session_file(self, filename):
        f = open(filename, 'r')
        self.session = pickle.load(f)
        f.close()
        self.get_from_session()

    def save_to_file(self, filename):
        """Saves self.info into the file.

        self.info must be a string with data.
        """
        f = open(filename, 'w')
        f.write(self.info)
        f.close()


class LearningCurve(Metric):
    """Learning curve of precision for the classifier.

    The information is a csv table separated by tabs where the first column
    represents the number of new instances or features the learner use to be
    trained, and the second column represents the precision of the classifier.

    The information is extracted from the recorded_precision element of the
    session.
    """

    def get_from_session(self):
        if not self.session or not 'recorded_precision' in self.session:
            self.info = ''
            return
        recorded_precision = self.session['recorded_precision']
        result = ''
        total_labels = 0
        for value in recorded_precision:
            total_labels += value['new_features'] + value['new_instances']
            result += '{}\t{}\n'.format(total_labels,
                                        value['testing_precision'])
        self.info = result


class PrecisionRecall(Metric):
    """Precision and recall for each of the classes for the last classifier.

    The information is a csv table separated by tabs where the content of each
    column is:
        class_name  precision   recall  F1_score

    The information is extracted from the classification_report element of
    the session.
    """

    def get_from_session(self):
        if not self.session or not 'classification_report' in self.session:
            self.info = ''
            return
        report = self.session['classification_report']
        result = []
        result = ['\t'.join(line.split()[:-1])
                  for line in report.split('\n')[3:-3]]
        self.info = '\n'.join(result)
