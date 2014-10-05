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
        if features:
            self.alpha = features
        return super(FeatMultinomalNB, self).fit(X, Y, sample_weight)
