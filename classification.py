#!/home/User/Miniconda3/python
import features, nltk, random, sys
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from nltk.classify import ClassifierI
from statistics import mode

f_train = open("C:/cygwin/home/User/nlp/project/2013-twiter-polarity-train.tsv.res", 'r', encoding='utf-8')
f_new = open("C:/cygwin/home/User/nlp/project/2013unlabeled", 'r', encoding='utf-8', errors = 'ignore')
results = open("C:/cygwin/home/User/nlp/project/scripts/results", 'a', encoding='utf-8')

l_classifiers = ['NB', 'Maxent', 'MNB', 'BernoulliNB', 'LogRegr', 'SGD', 'SVC', 'LinearSVC', 'RandomForest']
FOLDS = 1

def train_save(f_train, f_new, classifier, method, feature_set, n_words, n_bg):
	print(classifier)
	if('NB' == classifier):
		CLASSIFIER = nltk.classify.NaiveBayesClassifier
		def train_function(v_train):
			return CLASSIFIER.train(v_train)

	elif('Maxent' == classifier):
		CLASSIFIER = nltk.classify.MaxentClassifier
		def train_function(v_train):
			# ALGORITHMS = ['GIS', 'IIS', 'MEGAM', 'TADM']
			return CLASSIFIER.train(v_train, algorithm='IIS', max_iter=10)
	
	elif('MNB' == classifier):
		CLASSIFIER = SklearnClassifier(MultinomialNB())
		CLASSIFIER._vectorizer.sort = False
		def train_function(v_train):
			return CLASSIFIER.train(v_train)

	elif('BernoulliNB' == classifier):
		CLASSIFIER = SklearnClassifier(BernoulliNB())
		CLASSIFIER._vectorizer.sort = False
		def train_function(v_train):
			return CLASSIFIER.train(v_train)

	elif('LogRegr' == classifier):
		CLASSIFIER = SklearnClassifier(LogisticRegression())
		CLASSIFIER._vectorizer.sort = False
		def train_function(v_train):
			return CLASSIFIER.train(v_train)

	elif('SGD' == classifier):
		CLASSIFIER = SklearnClassifier(SGDClassifier())
		CLASSIFIER._vectorizer.sort = False
		def train_function(v_train):
			return CLASSIFIER.train(v_train)

	elif('SVC' == classifier):
		CLASSIFIER = SklearnClassifier(SVC())
		CLASSIFIER._vectorizer.sort = False
		def train_function(v_train):
			return CLASSIFIER.train(v_train)

	elif('LinearSVC' == classifier):
		CLASSIFIER = SklearnClassifier(LinearSVC())
		CLASSIFIER._vectorizer.sort = False
		def train_function(v_train):
			return CLASSIFIER.train(v_train)

	elif('RandomForest' == classifier):
		CLASSIFIER = SklearnClassifier(LinearSVC())
		CLASSIFIER._vectorizer.sort = False
		def train_function(v_train):
			return CLASSIFIER.train(v_train)

	def out_results_step1(v_new, classifier_tot):
		out = open("C:/cygwin/home/User/nlp/project/res", 'w', encoding='utf-8')
		for feat, label in v_new:
			label = classifier_tot.classify(feat)
			out.write(label + '\n')
		out.close()

	def out_results_step2(v_test_obj, classifier_obj, classifier_sen):
		out = open("C:/cygwin/home/User/nlp/project/res", 'w', encoding='utf-8')
		for feat, label in v_test_obj:
				label = classifier_obj.classify(feat)
				if label == 'obj':
					label = classifier_sen.classify(feat)
				out.write(label + '\n')
		out.close()

	if '1step' == method:
		for k in range(FOLDS):
			v_train, v_test, v_new = features.feature_extractor(f_train, f_new, FOLDS, k, method, feature_set, n_words, n_bg)
			classifier_tot = train_function(v_train)
		out_results_step1(v_new, classifier_tot)
		acc = nltk.classify.accuracy(classifier_tot, v_test)

	elif '2step' == method:
		for k in range(FOLDS):
			v_train_obj, v_train_sen, v_test_obj, v_test_sen, v_new = features.feature_extractor(f_train, f_new, FOLDS, k, method, feature_set, n_words, n_bg)
			classifier_obj = train_function(v_train_obj)
			classifier_sen = train_function(v_train_sen)
			accuracy_obj = nltk.classify.accuracy(classifier_obj, v_test_obj)
			accuracy_sen = nltk.classify.accuracy(classifier_sen, v_test_sen)

			classifier_tot = (classifier_obj, classifier_sen)
			#out_results_step2(v_test_obj, classifier_obj, classifier_sen)
	
		out_results_step2(v_new, classifier_obj, classifier_sen)
	results.write('%s\t%s\t%s\t%s\t%s\t%s\t%s\t\t%s\n' % (classifier, method, feature_set['ngram'], feature_set['negtn'], feature_set['best'], n_words, n_bg, acc))
	return classifier_tot

 
class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self._classifiers = classifiers

	def classify(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		return mode(votes)

	def confidence(self, features):
		votes = []
		for c in self._classifiers:
			v = c.classify(features)
			votes.append(v)
		choice_votes = votes.count(mode(votes))
		conf = choice_votes / len(votes)
		return conf



# n_words - number of the most popular words
# n_bg - number of the most popular bigrams
train_save(f_train=f_train, f_new=f_new, classifier=cname, method='1step', feature_set={'ngram':ngramVal, 'negtn':negtnVal, 'best':True}, n_words=780, n_bg=6)



FOLDS = 10

random.shuffle(l_sent)

for k in range(FOLDS):
   training = [x for i, x in enumerate(l_sent) if i % K != k]
   validation = [x for i, x in enumerate(l_sent) if i % K == k]