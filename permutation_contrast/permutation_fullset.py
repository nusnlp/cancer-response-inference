import json
import copy
import random
import itertools
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

from utils import STOPWORDS, clean, dl_clean

random.seed(0)

def read_json(d):
	with open(d, 'r', encoding='utf-8') as f:
		return json.load(f)

def write_json(data, dsave):
    outf = open(dsave, 'w', encoding='utf-8')
    json.dump(data, outf, ensure_ascii=False)
    outf.close()
    print('>>> write to {}'.format(dsave))

def write_jsonl(data, dsave):
	outf = open(dsave, 'w')
	for d in data:
		json.dump(d, outf)
		outf.write('\n')
	outf.close()
	print('\n+++ write to {}'.format(dsave))

def clean_report(text):
    if 'findings' in text:
        pos_s = text.index('findings') + len('findings')
        # print(len(combined_text), len(combined_text[pos_s:].strip()))
        text = text[pos_s:].strip()
        if len(text) > 1 and text[0] == ':':
            text = text[1:].strip()
    return text

label_count = [0] * 4
dir_data = '/path/to/train.json'
dataw = read_json(dir_data)
generated = []

insts_all = []
N_copies = 11

sent_length = 0

for d in tqdm(dataw['data'], ncols=100):
    label = d['label']
    ctext = d['conclusion']
    rtext = d['report']

    insts = []
    csents = sent_tokenize(ctext)
    sent_length += len(csents)
    pos = 6
    if len(csents) > pos:
        perms = list(itertools.permutations(csents[:pos]))
        perms = [list(p) + csents[pos:] for p in perms]
    else:
        perms = list(itertools.permutations(csents))
    if len(perms) > N_copies:
        perms = perms[:N_copies]
    else:
        perms = (perms*N_copies)[:N_copies]

    # REPORT
    rtext = clean_report(rtext)
    rsents = sent_tokenize(rtext)
    if len(rsents) > pos:
        rperms = list(itertools.permutations(rsents[:pos]))
        rperms = [list(p) + rsents[pos:] for p in rperms]
    else:
        rperms = list(itertools.permutations(rsents))
    if len(rperms) > N_copies:
        rperms = rperms[:N_copies]
    else:
        rperms = (rperms*N_copies)[:N_copies]

    # for s in perms:
    for i in range(N_copies):
        dp = copy.deepcopy(d)
        dp['conclusion'] = ' '.join(perms[i])
        dp['report'] = ' '.join(rperms[i])
        insts.append(dp)
    label_count[label] += len(perms)
    insts_all.append(insts)

combined = []
for i in range(N_copies):
    combined += [s[i] for s in insts_all]

dataw['data'] = combined

write_json(dataw, './permuted_data/trainS_x{}.json'.format(N_copies))
