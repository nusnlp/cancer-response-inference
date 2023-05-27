"""
	Example usage: 
	python t_test.py -base base_preds.jsonl \
		-pet pet_preds.json \
		-y gold.jsonl
"""

from scipy import stats
import os
import ast
import argparse
import random
import numpy as np
from sklearn.metrics import accuracy_score



def fix_seed():
	""" Enable reproducibility """
	random.seed(123)
	os.environ['PYTHONHASHSEED'] = str(123)

def load_jsonl_base(res_fp):
	a = open(res_fp).readlines()
	a = [ast.literal_eval(x) for x in a]
	return [str(x['label_pred']) for x in a]

def load_jsonl_pet(res_fp):
	a = open(res_fp).readlines()
	a = [ast.literal_eval(x) for x in a]
	return [x['label'] for x in a]

def load_jsonl_gold(res_fp):
	a = open(res_fp).readlines()
	a = [ast.literal_eval(x) for x in a]
	return [x['LBL'] for x in a]

def t_test(args):
	# Get test y
	gold = load_jsonl_gold(args.test_gold_fp)
	
	# Get base model test preds
	pet_preds = load_jsonl_pet(args.pet_test_preds_fp)

	# Get pet test preds
	base_preds = load_jsonl_base(args.base_test_preds_fp)

	# Randomly sample 20% of test results for base and pet respectively,
	# repeat for 200 times
	all_base_acc = []
	all_pet_acc = []
	len_test = len(gold)
	sample_size = int(len_test*0.2)
	for i in range(200):
		sample_inds = random.sample(range(0, len_test), sample_size)
		# print('==={}==='.format(i))
		# print(sample_inds[:5])
		test_y = np.array(gold)[sample_inds]
		base_test_preds = np.array(base_preds)[sample_inds]
		pet_test_preds = np.array(pet_preds)[sample_inds]
		base_acc = accuracy_score(test_y, base_test_preds)
		pet_acc = accuracy_score(test_y, pet_test_preds)
		all_base_acc.append(base_acc)
		all_pet_acc.append(pet_acc)
		
		# Get p-value
	p_val = stats.ttest_ind(all_base_acc, all_pet_acc, equal_var=False)
	print('p value: ', p_val)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-y', "--test_gold_fp", required=True)
	parser.add_argument("-base", "--base_test_preds_fp", required=True)
	parser.add_argument("-pet", "--pet_test_preds_fp", required=True)

	args = parser.parse_args()
	fix_seed()

	t_test(args)











