#clustering metrics 
"""
ACC,NMI,ARI,
Pair confusion Matrix,Silhouette Coefficient,
Fowlkes-Mallows scores,V-measure = (homogenety completeness)

"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics import homogeneity_score as homo_score
from sklearn.metrics import completeness_score as com_score
from sklearn.metrics import fowlkes_mallows_score as fmi_score
from sklearn.metrics import silhouette_score as silh_score
from sklearn.metrics.cluster import contingency_matrix



def pair_confusion_matrix(labels_true, labels_pred):
    """Pair confusion matrix arising from two clusterings.
    The pair confusion matrix :math:`C` computes a 2 by 2 similarity matrix
    between two clusterings by considering all pairs of samples and counting
    pairs that are assigned into the same or into different clusters under
    the true and predicted clusterings.
    Considering a pair of samples that is clustered together a positive pair,
    then as in binary classification the count of true negatives is
    :math:`C_{00}`, false negatives is :math:`C_{10}`, true positives is
    :math:`C_{11}` and false positives is :math:`C_{01}`.
    Read more in the :ref:`User Guide <pair_confusion_matrix>`.
    Parameters
    ----------
    labels_true : array-like of shape (n_samples,), dtype=integral
        Ground truth class labels to be used as a reference.
    labels_pred : array-like of shape (n_samples,), dtype=integral
        Cluster labels to evaluate.
    Returns
    -------
    C : ndarray of shape (2, 2), dtype=np.int64
        The contingency matrix.
    See Also
    --------
    rand_score: Rand Score
    adjusted_rand_score: Adjusted Rand Score
    adjusted_mutual_info_score: Adjusted Mutual Information
    Examples
    --------
    Perfectly matching labelings have all non-zero entries on the
    diagonal regardless of actual label values:
      >>> from sklearn.metrics.cluster import pair_confusion_matrix
      >>> pair_confusion_matrix([0, 0, 1, 1], [1, 1, 0, 0])
      array([[8, 0],
             [0, 4]]...
    Labelings that assign all classes members to the same clusters
    are complete but may be not always pure, hence penalized, and
    have some off-diagonal non-zero entries:
      >>> pair_confusion_matrix([0, 0, 1, 2], [0, 0, 1, 1])
      array([[8, 2],
             [0, 2]]...
    Note that the matrix is not symmetric.
    References
    ----------
    .. L. Hubert and P. Arabie, Comparing Partitions, Journal of
      Classification 1985
      https://link.springer.com/article/10.1007%2FBF01908075
    """

    n_samples = np.int64(labels_true.shape[0])

    # Computation using the contingency data
    contingency = contingency_matrix(
        labels_true, labels_pred, sparse=True
    )
    n_c = np.ravel(contingency.sum(axis=1))
    n_k = np.ravel(contingency.sum(axis=0))
    sum_squares = (contingency.data ** 2).sum()
    C = np.empty((2, 2), dtype=np.int64)
    C[1, 1] = sum_squares - n_samples
    C[0, 1] = contingency.dot(n_k).sum() - sum_squares
    C[1, 0] = contingency.transpose().dot(n_c).sum() - sum_squares
    C[0, 0] = n_samples ** 2 - C[0, 1] - C[1, 0] - sum_squares
    return C






def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    # from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size





class all_metrics():
    def __init__(self,latent,y,n_clusters,n_init,n_jobs):
        self.kmeans = KMeans(n_clusters=n_clusters,n_init=n_init,n_jobs=n_jobs)
        self.y_pred = self.kmeans.fit_predict(latent)
        self.y = y
        self.latent = latent

    def scores(self):
        nmi = nmi_score(self.y,self.y_pred)
        acc = cluster_acc(self.y,self.y_pred)
        ari = ari_score(self.y,self.y_pred)

        # silhoutte_co = silh_score(self.latent,self.y_pred,metric='euclidean')
        fmi = fmi_score(self.y,self.y_pred)

        homo = homo_score(self.y,self.y_pred)
        completeness = com_score(self.y,self.y_pred)

        pcm = pair_confusion_matrix(self.y,self.y_pred)
        C00 = pcm[0][0]
        C01 = pcm[0][1]
        C10 = pcm[1][0]
        C11 = pcm[1][1]

        # return {"nmi":nmi,"acc":acc,"ari":ari,"fmi":fmi,"homogenety":homo,"completeness":completeness,"pair_confusion_matrix":pcm,"silhoutte_co":silhoutte_co}
        return {"nmi":nmi,"acc":acc,"ari":ari,"fmi":fmi,"homogenety":homo,"completeness":completeness,"C00":C00,"C01":C01,"C10":C10,"C11":C11}