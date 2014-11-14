import numpy as np
from math import log
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils.extmath import safe_sparse_dot

def _sanitize_logarithms(X):
    """Changes the nan
    """


class FeatMultinomalNB(MultinomialNB):
    """A MultinomialNB classfier that can be trained using labeled features.
    """

    def fit(self, X, Y, sample_weight=None, features=None):
        """Fit Naive Bayes classifier according to X, y

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        sample_weight : array-like, shape = [n_samples], optional
            Weights applied to individual samples (1. for unweighted).

        features : array-like, shape = [n_classes, n_features], optional
            Boost for the prior probability of a feature given a class. For no
            boost use the value alpha given on the initialization.

        Returns
        -------
        self : object
            Returns self.
        """
        if features != None:
            self.alpha = features
        self.instance_num = X.shape[0]
        return_value = super(FeatMultinomalNB, self).fit(X, Y, sample_weight)
        self._information_gain()
        return return_value

    def _count(self, X, Y):
        super(FeatMultinomalNB, self)._count(X, Y)
        # Number of intances with class j and precense of feature k
        # shape class, feat.
        self.count_feat_and_class = safe_sparse_dot(Y.T, (X > 0))

    def _information_gain(self):
        """Calculates the information gain for each feature.

        Stores the value in self.feat_information_gain
        """
        prob_Ik1_and_class = self.count_feat_and_class / self.instance_num
        prob_Ik1 = self.count_feat_and_class.sum(axis=0) / self.instance_num
        term1 = (prob_Ik1_and_class *
                 ((np.log(prob_Ik1_and_class) - np.log(prob_Ik1)).T -
                  self.class_log_prior_).T)
        prob_Ik0_and_class = ((self.count_feat_and_class.T -
                               self.class_count_).T * -1 / self.instance_num)
        term2 = (prob_Ik0_and_class *
                 ((np.log(prob_Ik0_and_class) - np.log(1 - prob_Ik1)).T -
                  self.class_log_prior_).T)
        self.feat_information_gain = (np.nan_to_num(term1).sum(axis=0) +
                                      np.nan_to_num(term2).sum(axis=0))

    def instance_proba(self, X):
        """Calculates the probability of each instance in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]

        Returns
        -------
        array-like, shape = [n_samples]
        """
        feat_prob = safe_sparse_dot(np.exp(self.class_log_prior_),
                                    np.exp(self.feature_log_prob_)).T
        instance_log_prob = safe_sparse_dot(X, np.log(feat_prob))
        return np.exp(instance_log_prob)
