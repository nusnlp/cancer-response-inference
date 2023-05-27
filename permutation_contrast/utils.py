import re
import string
from nltk import word_tokenize
import torch
import torch.nn.functional as F

STOPWORDS = \
	["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
	 "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
	 "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
	 "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
	 "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
	 "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
	 "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there",
	 "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
	 "such", "only", "own", "same", "so", "than", "too", "very", "s", "t",
	 "can", "will", "just", "don", "should", "now"]

def clean(text):
	text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
	text = re.sub(r"what's", "what is ", text)
	text = re.sub(r"\'s", " ", text)
	text = re.sub(r"\'ve", " have ", text)
	text = re.sub(r"can't", "cannot ", text)
	text = re.sub(r"n't", " not ", text)
	text = re.sub(r"i'm", "i am ", text)
	text = re.sub(r"\'re", " are ", text)
	text = re.sub(r"\'d", " would ", text)
	text = re.sub(r"\'ll", " will ", text)
	text = re.sub(r",", " ", text)
	# text = re.sub(r"\.", " ", text)
	text = re.sub(r"!", " ! ", text)
	# text = re.sub(r"\/", " ", text)
	text = re.sub(r"\^", " ^ ", text)
	text = re.sub(r"\+", " + ", text)
	# text = re.sub(r"\-", " - ", text)
	text = re.sub(r"\=", " = ", text)
	text = re.sub(r"'", " ", text)
	text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
	text = re.sub(r":", " : ", text)
	# text = re.sub(r" e g ", " eg ", text)
	# text = re.sub(r" b g ", " bg ", text)
	# text = re.sub(r" u s ", " american ", text)
	text = re.sub(r"\0s", "0", text)
	# text = re.sub(r" 9 11 ", "911", text)
	text = re.sub(r"e - mail", "email", text)
	text = re.sub(r"j k", "jk", text)
	text = re.sub(r"\s{2,}", " ", text)
	text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
	# text = re.sub(r'(\-\s)(\-\s)+', '', text)
	text = re.sub(r'(\-)(\-)+', '', text)
	# return text.lower()
	return text

def dl_clean(text):
	text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
	text = re.sub(r"what's", "what is ", text)
	text = re.sub(r"\'s", " ", text)
	text = re.sub(r"\'ve", " have ", text)
	text = re.sub(r"can't", "cannot ", text)
	text = re.sub(r"n't", " not ", text)
	text = re.sub(r"i'm", "i am ", text)
	text = re.sub(r"\'re", " are ", text)
	text = re.sub(r"\'d", " would ", text)
	text = re.sub(r"\'ll", " will ", text)
	text = re.sub(r",", " ", text)
	text = re.sub(r"\.", " ", text)
	text = re.sub(r"!", " ! ", text)
	text = re.sub(r"\/", " ", text)
	text = re.sub(r"\^", " ^ ", text)
	text = re.sub(r"\+", " + ", text)
	text = re.sub(r"\-", " - ", text)
	text = re.sub(r"\=", " = ", text)
	text = re.sub(r"'", " ", text)
	text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
	text = re.sub(r":", " : ", text)
	text = re.sub(r"\0s", "0", text)
	text = re.sub(r" 9 11 ", "911", text)
	text = re.sub(r"e - mail", "email", text)
	text = re.sub(r"j k", "jk", text)
	text = re.sub(r"\s{2,}", " ", text)
	# return text.lower()
	return text

def get_text_features(text):
	text = text.lower()
	N_features = 63
	feature_vector = [0] * N_features
	words = word_tokenize(text)
	words = [w.strip() for w in words if w not in string.punctuation]
	if words[0] == 'conclusion':
		words = words[1:]
	# 0: length of text
	feature_vector[0] = len(words)
	# feature_vector[0] = len(words)/256
	# 1, 2: evidence, no evidence
	if 'evidence' in words:
		feature_vector[1] = 1
		if 'no evidence' in text:
			feature_vector[2] = 1
	# 3: further action
	if 'further action' in text:
		feature_vector[3] = 1
	# 4, 5: increase, increase in size
	if 'increase' in words:
		feature_vector[4] = 1
	if 'increase in size' in text:
		feature_vector[5] = 1
	# 6: decrease
	if 'decrease' in text:
		feature_vector[6] = 1
	# 7: detected
	if 'detected' in text:
		feature_vector[7] = 1
	# 8: stable
	if 'stable' in text:
		feature_vector[8] = 1
	# 9: recurrence
	if 'recurrence' in text:
		feature_vector[9] = 1
	# 10: small
	if 'small' in text:
		feature_vector[10] = 1
	# 11: size
	if 'size' in text:
		feature_vector[11] = 1
	# 12, 13: normal, abnormal
	if 'normal' in words:
		feature_vector[12] = 1
	if 'abnormal' in words:
		feature_vector[13] = 1
	if 'compared' in words:
		feature_vector[27] = 1
	if 'suspicious' in words:
		feature_vector[29] = 1
	if 'liver' in words:
		feature_vector[31] = 1
	if 'lung' in words:
		feature_vector[33] = 1
	if 'kidney' in words:
		feature_vector[35] = 1
	if 'abdomen' in words:
		feature_vector[37] = 1
	if 'pulmonary' in text:
		feature_vector[39] = 1
	if 'bone' in words:
		feature_vector[41] = 1
	if 'spine' in words:
		feature_vector[43] = 1
	if 'lymph' in text:
		feature_vector[45] = 1
	if 'ovary' in text:
		feature_vector[47] = 1
	if 'eum' in text:
		feature_vector[49] = 1
	if 'skin' in words:
		feature_vector[51] = 1
	if 'brain' in words:
		feature_vector[53] = 1
	if 'pleura' in text:
		feature_vector[55] = 1
	if 'spleen' in words:
		feature_vector[57] = 1
	if 'no' in words:
		feature_vector[59] = 1
	if 'change' in words:
		feature_vector[61] = 1

	# count
	feature_vector[14] = words.count('evidence')
	feature_vector[15] = text.count('no evidence')
	feature_vector[16] = text.count('further action')
	feature_vector[17] = words.count('increase')
	feature_vector[18] = text.count('increase in size')
	feature_vector[19] = text.count('decrease')
	feature_vector[20] = text.count('detected')
	feature_vector[21] = text.count('stable')
	feature_vector[22] = text.count('recurrence')
	feature_vector[23] = text.count('small')
	feature_vector[24] = text.count('size')
	feature_vector[25] = words.count('normal')
	feature_vector[26] = words.count('abnormal')
	feature_vector[28] = words.count('compared')
	feature_vector[30] = words.count('suspicious')
	feature_vector[32] = words.count('liver')
	feature_vector[34] = words.count('lung')
	feature_vector[36] = words.count('kidney')
	feature_vector[38] = words.count('abdomen')
	feature_vector[40] = words.count('pulmonary')
	feature_vector[42] = words.count('bone')
	feature_vector[44] = words.count('spine')
	feature_vector[46] = text.count('lymph')
	feature_vector[48] = text.count('ovary')
	feature_vector[50] = text.count('eum')
	feature_vector[52] = words.count('skin')
	feature_vector[54] = words.count('brain')
	feature_vector[56] = words.count('pleura')
	feature_vector[58] = words.count('spleen')
	feature_vector[60] = words.count('no')
	feature_vector[62] = words.count('change')

	return feature_vector

def soft_cross_entropy(input, target, reduction='mean'):
	nll = -F.log_softmax(input, dim=1)
	bsz = input.shape[0]
	loss = torch.sum(torch.mul(nll, target))
	if reduction == 'mean':
		loss = loss/bsz
	return loss

def mimic_clean(text):
	text = clean(text)
	text = text.lower()
	text = text.replace(' technique :', '. technique :')
	text = text.replace(' comparison :', '. comparison :')
	text = text.replace(' comparisons :', '. comparison :')
	text = text.replace(' findings :', '. findings :')
	text = text.replace(' finding :', '. findings :')
	text = text.replace(' history :', '. history :')
	text = text.replace(' impression :', '. impression :')
	text = text.replace(' impressions :', '. impression :')
	text = text.replace(' indication :', '. indication :')
	text = text.replace(' indications :', '. indication :')
	text = text.replace(' examination :', '. examination :')
	ext = text.replace(' exam :', '. examination :')

	return text