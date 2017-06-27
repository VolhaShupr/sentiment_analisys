#!/home/User/Miniconda3/python

# The module is used for preprocessing text

import re, nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

def tokenizer(sent):
	l_sent = word_tokenize(sent)
	return l_sent

def del_proper_words(sent):
	el = re.findall(r"[\.!\?]+ *[A-Z][A-Za-z]+", sent)
	for i in el:
		if el:
			sent = sent.replace(i, i.lower())
	sent = re.sub(r"\t( +)", '\t', sent)
	sent = re.sub(r" [A-Z][a-z]+", '', sent)
	return sent

def repeated_letters(sent):
	matches = re.finditer(r"([a-z])\1{2,}", sent)
	if matches:
		for m in matches:	
			sent = re.sub( m.group(), m.group(1), sent)
	return sent

def spelling(sent):
	with open("english_words.txt") as word_file:
		english_words = set(word.strip().lower() for word in word_file)
	tok_sent = tokenizer(sent)
	l = []
	for word in tok_sent:
		if word in english_words:
			l.append(word)
	s = ' '.join(l)
	return s		 

def if_negation(word):
	NEG = ["n't", 'any', "arent", "cannot", "cant", "couldnt", "darent", "didnt", "doesnt", "dont", "hadnt", "hasnt", "havent", "isnt", "mightnt", "mustnt", "neither", "neednt", "none", "noone", "nope", "nor", "nothing", "nowhere", "oughtnt", "shant", "shouldnt", "wasnt", "werent", "without", "wont", "wouldnt", "rarely", "seldom"]
	for i in NEG:
		if i == word:
			word = 'not'
	return word

def filtering(sent, prepos=True, lemm=True, stemm=False):
	ps = PorterStemmer()
	wnl = WordNetLemmatizer()
	line = sent.split("\t")
	sent = line[-1]
	TAGGS = ['NN', 'VB', 'JJ', 'RB']
	prep = 'IN'
	stop_words = set(stopwords.words('english'))
	tok_sent = tokenizer(sent)
	tagged_sent = nltk.pos_tag(tok_sent)
	l = []
	for tword in tagged_sent:
		word = if_negation(tword[0])
		if word =='no' or word == 'not':
			l.append(word)
		for tag in TAGGS:
			if len(word) > 2 and tword[1].startswith(tag) and not word in stop_words:
				if stemm:
					l.append(ps.stem(word))
				elif lemm:
					l.append(wnl.lemmatize(word))
				else:
					l.append(word)
		if prepos:
			if tword[1] == prep:
				l.append(word)
	s = ' '.join(l)
	if not s:
		s = 'neutral'
	if len(line) == 1:
		sent = s
	else:
		sent = line[0] + '\t' + s
	return sent

def del_noise(sent, labels=False, hashtags=True, urls=True, retweets=True, non_letters=True, proper=True, lower=True, repeated=True):
	sent = sent.strip().split("\t")
	if labels:
		sent = sent[-1]
	else:
		lab = sent[-2]
		if lab != 'negative' and lab != 'positive' and lab != 'neutral':
			lab = 'neutral'
		sent = lab + '\t' + sent[-1]
	if urls:
		sent = re.sub(r"https?:\/\/[^\s]*", '', sent)
	if hashtags:
		sent = re.sub(r"#[^\s]*", '', sent)
	if retweets:
		sent = re.sub(r"@[^\s]*", '', sent)
	if proper:
		test = del_proper_words(sent)
		if len(tokenizer(filtering(test, prepos=False))) > 1:
			sent = del_proper_words(sent)	
	if non_letters:
		sent = re.sub(r"[^a-zA-Z\s']", ' ', sent)
	if lower:
		sent = sent.lower()
	if repeated:
		sent = repeated_letters(sent)
	return sent

if __name__ == '__main__':
	file = open("C:/cygwin/home/User/nlp/project/twitter_polarity_test.tsv", 'r', encoding='utf-8')
	output = open("C:/cygwin/home/User/nlp/project/scripts/preproc", 'w', encoding='utf-8')
	for sent in file:
		sent = del_noise(sent, labels=True, hashtags=True, urls=True, non_letters=True, proper=True, lower=True, repeated=True)
		sent = (filtering(sent, prepos=False, lemm=True, stemm=False))
		output.write(str(sent) + '\n')
	file.close()
	output.close()
	