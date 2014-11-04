import numpy as np
from math import log
from sklearn.naive_bayes import MultinomialNB


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
        self.training_instances = X.shape[0]
        return_value = super(FeatMultinomalNB, self).fit(X, Y, sample_weight)
        self._information_gain()
        return return_value

    # def _update_feature_log_prob(self):

    def _information_gain(self):
        """Calculates the information gain for each feature.

        Stores the value in self.feat_information_gain

        Returns
        -------
        array-like, shape = [n_features]
        """
        # Agrego el +1 para eliminar los 0, asi es como se hace?
        # Probability of the presence of a feature and a class.
        feat_and_class_prob = (self.feature_count_) / (self.training_instances + 0.0)
        # Features present in a class
        feat_per_class = self.feature_count_ > 0
        # P(Ik)  -- Should we apply some smothing?
        Ik_log_prob = np.log(feat_per_class.sum(axis=0)) - log(len(self.classes_))
        aux = np.log(feat_and_class_prob) - Ik_log_prob  # Shape (n_class, n_feat)
        aux = aux.T - self.class_log_prior_  # Shape (n_feat, n_class)
        aux = feat_and_class_prob.T * aux

        feat_per_class = self.feature_count_ == 0
        # P(Ik)  -- Should we apply some smothing?
        Ik_log_prob = np.log(feat_per_class.sum(axis=0) / (len(self.classes_) + 0.0))
        # Agrego el +1 para eliminar los 0, asi es como se hace?
        aux2 = np.log(1 - feat_and_class_prob) - Ik_log_prob  # Shape (n_class, n_feat)
        aux2 = aux2.T - self.class_log_prior_  # Shape (n_feat, n_class)
        aux2 = (1 - feat_and_class_prob.T) * aux2

        self.feat_information_gain = aux.sum(axis=1) + aux2.sum(axis=1) # Shape (n_feat)
