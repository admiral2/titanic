'''
Useful posts about ROC that helped me grok it:

http://fastml.com/what-you-wanted-to-know-about-auc/
http://scikit-learn.org/stable/auto_examples/plot_roc.html
http://en.wikipedia.org/wiki/Receiver_operating_characteristic
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def generate_roc_curve(clf, X, y, survived_weight=1, plot=False, n_classes=5):
    """
    Generates an ROC curve and calculates the AUC
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    aucs = []
    for i in range(5):
        # shuffle and split training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)
        #weights = np.array([survived_weight if s == 1 else 1 for s in y_train])
        #clf.fit(X_train, y_train, sample_weight=weights)
        clf.fit(X_train, y_train)

        fpr[i], tpr[i], _ = roc_curve(y_test, clf.predict_proba(X_test)[:,1])
        roc_auc[i] = auc(fpr[i], tpr[i])
        aucs.append(roc_auc[i])
        print('ROC AUC: {:.2%}'.format(roc_auc[i]))

    auc_mean = "{:.3%}".format(np.mean(aucs))
    auc_std = "{:.3%}".format(np.std(aucs))
    auc_lower = "{:.3%}".format(np.mean(aucs)-np.std(aucs))
    print("ROC - Area under curve: {} and stddev: {}".format(auc_mean, auc_std))

    if plot:
        # Plot of a ROC curve for a specific class
        plt.figure()
        for i in range(5):
            plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])

        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.show()
