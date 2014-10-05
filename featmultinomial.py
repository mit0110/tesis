from sklearn.naive_bayes import MultinomialNB


class FeatMultinomalNB(MultinomialNB):
    """A MultinomialNB classfier that can be trained using labeled features.
    """

    def fit(self, X, Y, a=None):
        super(FeatMultinomalNB, self).fit(X, Y, a)
