#!/home/User/Miniconda3/python
import re
from bs4 import BeautifulSoup

def unigrams(not_w=False):
	file = open("C:/cygwin/home/User/nlp/project/dicts/unigrams.so", "r")
	d_wd_so = {}
	if not_w:
		d_wd_so['not'] = 0.0
	for s_line in file:
		l_wd_so_pos_neg = s_line.strip().split("\t")
		if l_wd_so_pos_neg:
			s_wd = l_wd_so_pos_neg[0]
			f_so = float(l_wd_so_pos_neg[1])
			if s_wd not in d_wd_so:
				d_wd_so[s_wd] = f_so
	return d_wd_so

def afinn(not_w=False):
	file = open("C:/cygwin/home/User/nlp/project/dicts/afinn.txt", "r")
	d_wd_so = {}
	if not_w:
		d_wd_so['not'] = 0
	for s_line in file:
		l_wd_so_pos_neg = s_line.strip().split("\t")
		if l_wd_so_pos_neg:
			s_wd = l_wd_so_pos_neg[0]
			f_so = int(l_wd_so_pos_neg[1])
			if s_wd not in d_wd_so:
				d_wd_so[s_wd] = f_so
	return d_wd_so

def nrc(not_w=False):
	file = open("C:/cygwin/home/User/nlp/project/dicts/nrc-emotion.txt", "r")
	d_wd_so = {}
	d_wd_so['no'] = -1
	if not_w:
		d_wd_so['not'] = 0
	for s_line in file:
		l_wd_so_pos_neg = s_line.strip().split("\t")
		if l_wd_so_pos_neg:
			s_wd = l_wd_so_pos_neg[0]
			senti = l_wd_so_pos_neg[1]
			boolean = l_wd_so_pos_neg[2]
			if senti == 'positive' and  boolean == '1':
				d_wd_so[s_wd] = 1
			elif senti == 'negative' and  boolean == '1':
				d_wd_so[s_wd] = -1
	return d_wd_so

def semeval(not_w=False):
	file = open("C:/cygwin/home/User/nlp/project/dicts/semeval.txt", "r")
	d_wd_so = {}
	if not_w:
		d_wd_so['not'] = 0.0
	for s_line in file:
		s_line = re.sub(r"#", '', s_line)
		l_wd_so_pos_neg = s_line.strip().split("\t")
		if l_wd_so_pos_neg:
			s_wd = l_wd_so_pos_neg[1]
			f_so = float(l_wd_so_pos_neg[0])
			if s_wd not in d_wd_so:
				d_wd_so[s_wd] = f_so
	return d_wd_so

def senti_wn():
	file = open("C:/cygwin/home/User/nlp/project/dicts/sentiWN.txt", "r")
	d_wd_so = {}
	for s_line in file:
		s_line = re.sub(r"#", '\t', s_line)
		l_wd_so_pos_neg = s_line.strip().split("\t")
		if l_wd_so_pos_neg:
			s_wd = l_wd_so_pos_neg[4]
			pos = float(l_wd_so_pos_neg[2])
			neg = float(l_wd_so_pos_neg[3])
			if s_wd not in d_wd_so:
				if pos > 0 and neg == 0:
					d_wd_so[s_wd] = pos
				elif neg > 0 and pos == 0:
					d_wd_so[s_wd] = -neg		
	return d_wd_so

def hz(not_w=False):
	file = open("C:/cygwin/home/User/nlp/project/dicts/hz.txt", "r")
	d_wd_so = {}
	d_wd_so['no'] = -1
	if not_w:
		d_wd_so['not'] = 0
	for s_line in file:
		w = re.search(r"word1=([a-z]+)", s_line)
		s_wd = w.group(1)
		p = re.search(r"priorpolarity=([a-z]+)", s_line)
		f_so = p.group(1)
		if f_so == 'positive':
			d_wd_so[s_wd] = 1
		if f_so == 'negative':
			d_wd_so[s_wd] = -1
	return d_wd_so

if __name__ == '__main__':
	output = open("C:/cygwin/home/User/nlp/project/scripts/d.txt", "w")
	dct = unigrams()
	#dct = afinn()
	#dct = nrc()
	#dct = semeval()
	#dct = senti_wn()
	#dct = hz()
	for k, v in dct.items():
		output.write(k + '\t' + str(v) + '\n')