import pickle


class Metric(object):
    """Represents information and a method to obtained from a session file.

    To create a custom metric, redefine the get_from_session method that saves
    in self.info the content of the metric extracted from self.session.
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
        print self.info
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


def pr_from_confusion_matrix(cm):
    """Precision and recall for each class from the confusion_matrix.

    By definition, cm is such that cm[i][j] is the number of instances known
    to be in the class c_i but predicted in the class c_j.

    Args:
        cm: matrix like. Shape = (n_classes, n_classes). An sklearn confusion
        matrix for the classifier.

    Returns:
        A list of 2-uples where the fist value is the precision and the
        second is the recall.
    """
    result = []
    class_sum_v = cm.sum(axis=0)
    class_sum_h = cm.sum(axis=1)
    for class_row in range(cm.shape[0]):
        tp = cm[class_row][class_row]
        fp = class_sum_v[class_row] - tp
        fn = class_sum_h[class_row] - tp
        result.append((tp/float(tp+fp), tp/float(tp+fn)))
    return result


class PrecisionRecallCurve(Metric):
    """Precision and recall for each of the classes for each training.

    The information is a csv table separated by tabs where the content of each
    column is:
        class_name  number_of_instances precision   recall

    The information is extracted from the classification_report element of
    the session.
    """

    def get_from_session(self):
        self.info = ''
        if not (self.session and 'recorded_precision' in self.session
            and 'confusion_matrix' in self.session['recorded_precision'][-1]):
            return
        if not 'classes' in self.session:
            return
        result = []
        num_instances = 0
        classes = self.session['classes']
        for rec_p in self.session['recorded_precision']:
            cm = rec_p['confusion_matrix']
            prec_rec_class = pr_from_confusion_matrix(cm)
            num_instances += rec_p['new_instances'] + rec_p['new_features']
            result += ['\t'.join((classes[i], str(num_instances), str(p), str(r)))
                       for i, (p, r) in enumerate(prec_rec_class)]
        self.info = '\n'.join(result)


class KappaStatistic(Metric):
    """Calculates the Kappa Statistic for the last classifier.

    The information is a float number and it's obtained from the
    confusion_matrix element of the session.
    """

    def get_from_session(self):
        self.info = ''
        if not (self.session and 'recorded_precision' in self.session
            and 'confusion_matrix' in self.session['recorded_precision'][-1]):
            return
        # too hacky
        confusion_m = self.session['recorded_precision'][-1]['confusion_matrix']
        total_instances = float(confusion_m.sum())
        real_accuracy = confusion_m.diagonal().sum() / total_instances
        e_accuracy = (confusion_m.sum(axis=1) * confusion_m.sum(axis=0) /
                      total_instances)
        e_accuracy = e_accuracy.sum() / total_instances
        self.info = (real_accuracy - e_accuracy) / (1 - e_accuracy)
