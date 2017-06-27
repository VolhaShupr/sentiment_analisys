#!/home/User/Miniconda3/python
import nltk, nltk.classify.util, sys, re, numpy, random, dicts
from preproc import del_noise, filtering
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder

'''
def k_fold_cross_validation(X, K, randomise = False):
    """
    Generates K (training, validation) pairs from the items in X.
    Each pair is a partition of X, where validation is an iterable
    of length len(X)/K. So each training iterable is of length (K-1)*len(X)/K.
    If randomise is true, a copy of X is shuffled before partitioning,
    otherwise its order is preserved in training and validation.
    """
    if randomise: from random import shuffle; X=list(X); shuffle(X)
    for k in xrange(K):
        training = [x for i, x in enumerate(X) if i % K != k]
        validation = [x for i, x in enumerate(X) if i % K == k]
        yield training, validation
'''
FOLDS = 10
k = 0

def preprocessing(file, new=False):	
	l_sent = []  	# [(['match', 'tomorrow', 'busy', 'day', 'debate','debate'], 'neutral'), ...]
	all_words = []
	for line in file:
		if new:
			line = del_noise(line, labels=True, proper=True)
			line = filtering(line, prepos=False, lemm=True, stemm=False)
			line = line.strip()
			words = line.split(' ')
			label = 'label'
			#print(l_sent)
		else:
			line = del_noise(line, proper=True)
			line = filtering(line, prepos=False, lemm=True, stemm=False)
			line = line.strip().split("\t")
			label = line[0]
			words = line[1].split(' ')
		l_sent.append((words, label))
		for w in words:
			all_words.append(w)
	file.seek(0)
	return l_sent, all_words


def feature_extractor(f_train, f_new, K, k, method, feature_set, num_word, num_bg):
	add_ngram_feat = feature_set.get('ngram', 1)
	add_negtn_feat = feature_set.get('negtn', False)
	add_best_feat = feature_set.get('best', False)

	l_sent, all_w_train = preprocessing(f_train)
	#l_sent_train, all_w_train = preprocessing(f_train)
	l_sent_new, all_w_new = preprocessing(f_new, new=True)

	if K > 1:
		random.shuffle(l_sent)
		l_sent_train = [x for i,x in enumerate(l_sent) if i % K !=k]
		l_sent_test = [x for i,x in enumerate(l_sent) if i % K ==k]
	else:
		l_sent_train = l_sent
		l_sent_test = l_sent_new
	
	def l_pos_neg_nue_w(l_sent):
		# lists of pos, neg, nuet words
		words_pos = []
		words_neg = []
		words_neut = []
		for words, label in l_sent:
			for word in words:
				if label == 'positive':
					words_pos.append(word)
				elif label == 'negative':
					words_neg.append(word)
				else:
					words_neut.append(word)
		return words_pos, words_neg, words_neut

	def frequency(words_pos, words_neg, words_neut, num):
		# word overall frequency
		w_fd = nltk.FreqDist()
		# its frequency within each class
		cond_w_fd = nltk.ConditionalFreqDist()
		for word in words_pos:
			w_fd[word] += 1
			cond_w_fd['pos'][word] += 1
		for word in words_neg:
			w_fd[word] += 1
			cond_w_fd['neg'][word] += 1
		for word in words_neut:
			w_fd[word] += 1
			cond_w_fd['neut'][word] += 1
		pos_w_count = cond_w_fd['pos'].N()
		neg_w_count = cond_w_fd['neg'].N()
		neut_w_count = cond_w_fd['neut'].N()
		tot_w_count = pos_w_count + neg_w_count + neut_w_count
		
		word_scores = {}
		for word, freq in w_fd.items():
			pos_score = BigramAssocMeasures.chi_sq(cond_w_fd['pos'][word], (freq, pos_w_count), tot_w_count)
			neg_score = BigramAssocMeasures.chi_sq(cond_w_fd['neg'][word], (freq, neg_w_count), tot_w_count)
			neut_score = BigramAssocMeasures.chi_sq(cond_w_fd['neut'][word], (freq, neut_w_count), tot_w_count)
			word_scores[word] = pos_score + neg_score + neut_score
		# {'lunch': 1.4536422665231745, 'new': 1.676422816632956, 'spot': 5.212442490836022, ...}
		
		# get n the most popular words
		best_vals = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:num]
		best_words = set([w for w, s in best_vals])
		# {'gold', 'no', 'actually', 'edge', 'home'}
		return best_words

	words_pos, words_neg, words_neut = l_pos_neg_nue_w(l_sent_train)

	if add_best_feat:
		best_words = frequency(words_pos, words_neg, words_neut, num_word)

	else:
		best_words = all_w_train

	d_wd_so = dicts.afinn()
	######### features dicts:
	def full_dict_feats(words, num_bg):
		
		bag = {}
		words_uni = [word for word in words if word in best_words]
		for f in words_uni:
			if f in d_wd_so:
				bag[f] = d_wd_so[f]
		#result = bag
		if add_ngram_feat>=2 :
			score_fn=BigramAssocMeasures
			bigram_finder = BigramCollocationFinder.from_words(words)
			if add_best_feat:
				try:
					bigrams = bigram_finder.nbest(score_fn.chi_sq, num_bg)
					d = dict([(bigram, 1) for bigram in bigrams])
					d.update(bag)
					bag = d
				except ZeroDivisionError:
					bag = bag
			else:
				try:
					scored = bigram_finder.score_ngrams(score_fn.raw_freq)
					d = dict([(bigram, 1) for bigram, score in scored])
					d.update(bag)
					bag = d
				except ZeroDivisionError:
					bag = bag

		if add_ngram_feat>=3 :
			score_fn=TrigramAssocMeasures
			trigram_finder = TrigramCollocationFinder.from_words(words)
			if add_best_feat:
				try:
					trigrams = trigram_finder.nbest(score_fn.chi_sq, num_bg)
					d = dict([(trigram, 1) for trigram in trigrams])
					d.update(bag)
					bag = d
				except ZeroDivisionError:
					bag = bag
			else:
				try:
					scored = trigram_finder.score_ngrams(score_fn.raw_freq)
					d = dict([(trigram, 1) for trigram, score in scored])
					d.update(bag)
					bag = d
				except ZeroDivisionError:
					bag = bag

		return bag
		
	negtn_regex = re.compile( r"""\b(?:no|not)\b""", re.X)
	def get_negation_features(words):
		negtn = [ bool(negtn_regex.search(w)) for w in words ]
		left = [0.0] * len(words)
		prev = 0.0
		neg = False
		n = 0
		for i in range(0,len(words)):		
			if negtn[i]:
				neg = True
				n = 0
				left[i] = 0.0
			if words[i] in d_wd_so:
				if neg == False:
					left[i] = d_wd_so[words[i]]
				else:
					while n < 4 and neg:
						left[i] = -d_wd_so[words[i]]
						n += 1
						if n == 3:
							neg = False
						break
		return dict( zip([w for w in  words], left) )

	def extractor(words):
		features = {}
		word_features = full_dict_feats(words, num_bg)
		#print(word_features)
		features.update( word_features )
		if add_negtn_feat :
			negation_features = get_negation_features(words)
			features.update( negation_features )
		return features

	if( '1step' == method ):
		# Apply NLTK's Lazy Map
		v_train = nltk.classify.util.apply_features(extractor, l_sent_train)
		v_test  = nltk.classify.util.apply_features(extractor, l_sent_test)
		v_new  = nltk.classify.util.apply_features(extractor, l_sent_new)
		return (v_train, v_test, v_new)

	elif( '2step' == method ):
		isObj   = lambda sent: sent in ['negative','positive']
		makeObj = lambda sent: 'obj' if isObj(sent) else sent
		
		train_tweets_obj = [ (words, makeObj(sent)) for (words, sent) in l_sent_train ]
		test_tweets_obj  = [ (words, makeObj(sent)) for (words, sent) in l_sent_test ]

		train_tweets_sen = [ (words, sent) for (words, sent) in l_sent_train if isObj(sent) ]
		test_tweets_sen  = [ (words, sent) for (words, sent) in l_sent_test if isObj(sent) ]

		v_train_obj = nltk.classify.util.apply_features(extractor,train_tweets_obj)
		v_train_sen = nltk.classify.util.apply_features(extractor,train_tweets_sen)
		v_test_obj  = nltk.classify.util.apply_features(extractor,test_tweets_obj)
		v_test_sen  = nltk.classify.util.apply_features(extractor,test_tweets_sen)
		v_new  = nltk.classify.util.apply_features(extractor, l_sent_new)
		return v_train_obj,v_train_sen,v_test_obj,v_test_sen, v_new


if __name__ == '__main__':
	f_train = open("C:/cygwin/home/User/nlp/project/2013-twiter-polarity-train.tsv.res", 'r', encoding='utf-8')
	f_test = open("C:/cygwin/home/User/nlp/project/2013-twiter-polarity-dev.tsv.res", 'r', encoding='utf-8')

	def out_results_step1(v_test, classifier_tot):
		out = open("C:/cygwin/home/User/nlp/project/res", 'w', encoding='utf-8')
		for feat, label in v_test:
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

