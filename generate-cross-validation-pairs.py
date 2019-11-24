### Principal Component Analysis
# Dependencies
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier as BlackBoxClassifier

def generate_cv_pairs(self, n_samples, n_folds=5, n_rep=1, rand=False,
                          y=None):
    array = (np.array(self))
    k = n_folds
    width = (n_samples)//k
    train = []
    test = []
    train_class = []
    test_class = []
    iteration=0
    data_set=[]
    
    def my_shuffle(array):
        random.shuffle(array)
        return array

    def cross_validation(data, k):    
        for i in range(0,k):
            if i<k-1:
                low = width*i
                high = width*(i+1)
                train_class.append(data[low:high][:,-1])
                train.append(np.delete(data[low:high], -1, axis=1))
            else:
                test_class.append(data[low:high][:,-1])
                test.append(np.delete(data[low:high], -1, axis=1))
        return [np.concatenate(train), np.concatenate(train_class)], [np.concatenate(test), np.concatenate(test_class)]
    
    while iteration<n_rep:
        iteration=iteration+1
        shuffled_data = my_shuffle(array)
        data_set.append(cross_validation(shuffled_data, k))
    return data_set
    """ Train and test pairs according to k-fold cross validation

        Parameters
        ----------

        n_samples : int
            The number of samples in the dataset

        n_folds : int, optional (default: 5)
            The number of folds for the cross validation

        n_rep : int, optional (default: 1)
            The number of repetitions for the cross validation

        rand : boolean, optional (default: False)
            If True the data is randomly assigned to the folds. The order of the
            data is maintained otherwise. Note, *n_rep* > 1 has no effect if
            *random* is False.

        y : array-like, shape (n_samples), optional (default: None)
            If not None, cross validation is performed with stratification and
            y provides the labels of the data.

        Returns
        -------

        cv_splits : list of tuples, each tuple contains two arrays with indices
            The first array corresponds to the training data, the second to the
            testing data for the current split. The list has the length of
            *n_folds* x *n_rep*.

        """
