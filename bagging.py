#!/usr/bin/python
import sys
import itertools
import optparse
import os
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from pandas import read_csv, DataFrame


NAME = os.path.basename(sys.argv[0])


def fit_ensemble(attributes,class_val,n_estimators):
	
	# max depth is 1
	decisionStump = DecisionTreeClassifier(criterion = 'entropy', max_depth = 1)

	ensemble = BaggingClassifier(base_estimator = decisionStump, n_estimators = n_estimators)
	return ensemble.fit(attributes,class_val)

def predict_all(fitted_classifier, instances):
	for i, instance in enumerate(instances):
		instances[i] = fitted_classifier.predict([instances[i]])
	return list(itertools.chain(*instances))

def main(filename, n_estimators):

	df_ = read_csv(filename)

	col_names = df_.columns.values.tolist()
	attributes = col_names[0:-1] ## 0..n-1
	class_val = col_names[-1] ## n

	fitted = fit_ensemble(df_[attributes].values, df_[class_val].values, n_estimators)
	fitted_classifiers = fitted.estimators_ # get the three decision stumps.

	compared_ = DataFrame(index = range(0,len(df_.index)), columns = range(0,n_estimators + 1))
	compared_ = compared_.fillna(0)
	compared_.ix[:,n_estimators] = df_[class_val].values

	for i, fitted_classifier in enumerate(fitted_classifiers):
		compared_.ix[:,i] = predict_all(fitted_classifier,df_[attributes].values)

	subsets = fitted.estimators_samples_
	print(compared_)
	
	for i, subset in enumerate(subsets):
		print("Decision stump " + str(i) + " subset: ")
		subsets[i] = list(itertools.chain(*df_[attributes].values[subset]))
		print(subsets[i])

	#subsets = DataFrame(subsets,index=subsets[0,1:],columns=list([1,2,3]))
	#print(subsets)


if __name__ == "__main__":
	try:
		filename = sys.argv[1]
	except IndexError:
		print "usage: python " + NAME + " <filename.csv>"
		sys.exit(1)

	usage = "usage: ./%s foldername [options]" % NAME

	parser = optparse.OptionParser(usage = usage)
	parser.add_option('-n','--n_estimators',type="int",action="store",
		dest="n_estimators",help="number of decision stumps to train",default=3)

	(options, args) = parser.parse_args()

	main(filename, options.n_estimators)

