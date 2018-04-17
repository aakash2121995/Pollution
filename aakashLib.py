import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats.stats import pearsonr
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error as mae
def drawCorr(df):
	corr = df.select_dtypes(include = ['float64', 'int64','uint8','float32']).iloc[:, 0:].corr()
	plt.figure(figsize=(30,30))
	sns.heatmap(corr, vmax=1, square=True)

def showCorr(df,col):
	corr = df.select_dtypes(include = ['float64', 'int64','float32']).iloc[:, 0:].corr()
	cor_dict = corr[col].to_dict()
	del cor_dict[col]
	print("List the numerical features decendingly by their correlation with:" + str(col))
	for ele in sorted(cor_dict.items(), key = lambda x: -abs(x[1])):
	    print("{0}: \t{1}".format(*ele))

def plotMissingVals(df,cols=[]):
	labels = []
	values = []
	returnData = {}
	for col in df.columns:
	    labels.append(col)
	    nulls = df[col].isnull().sum()
	    values.append(nulls)
	    for col_ in cols:
	    	if col.find(col_) >= 0:
	    		returnData[col_] = nulls
	    print(col, values[-1])
	ind = np.arange(len(labels))
	width = 0.9
	fig, ax = plt.subplots(figsize=(12,50))
	rects = ax.barh(ind, np.array(values), color='y')
	ax.set_yticks(ind+((width)/2.))
	ax.set_yticklabels(labels, rotation='horizontal')
	ax.set_xlabel("Count of missing values")
	ax.set_title("Number of missing values in each column")
	plt.show()
	return returnData

def showSkew(df):
	numeric_feats = df.dtypes[df.dtypes != "object"].index
	skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna()))#compute skewness
	print (skewed_feats)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def run_kfold(model,X,y,folds=5):
	kf = KFold(X.shape[0], n_folds=folds)
	outcomes = []
	fold = 0
	for train_index, test_index in kf:
		fold += 1
		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y.values[train_index], y.values[test_index]
		model.fit(X_train, y_train)
		predictions = (model.predict(X_test))
		accuracy = mae(y_test, predictions)
		outcomes.append(accuracy)
		print("Fold {0} accuracy: {1}".format(fold, accuracy))     
		mean_outcome = np.mean(outcomes)
	print("Mean Accuracy: {0}".format(mean_outcome))


