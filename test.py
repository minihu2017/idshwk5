import sklearn
import numpy as np
import math
from sklearn.ensemble import RandomForestClassifier

Domainlist = []

class DomainClass:
	def __init__(self, domain, lable = None):
		self.name = domain
		self.lable = lable
	
	def GetLen(self):
		return len(self.name)

	def GetEncropy(self):
		tmp_dict = {}
		domain_len = len(self.name)
		for i in range(0,domain_len):
			if self.name[i] in tmp_dict.keys():
				tmp_dict[self.name[i]] = tmp_dict[self.name[i]] + 1
			else:
				tmp_dict[self.name[i]] = 1
		Entropy = 0
		for i in tmp_dict.keys():
			p = float(tmp_dict[i]) / domain_len
			Entropy = Entropy - p * math.log(p,2)
		return Entropy

	def GetVowel(self):
		Vowel = ['a','e','i','o','u']
		self.name = self.name.lower()
		count_word = 0
		count_vowel = 0
		vowel_ratio = 0
		for i in range(0,len(self.name)):
			if ord(self.name[i]) >= ord('a') and ord(self.name[i]) <= ord('z'):
				count_word = count_word + 1
				if self.name[i] in Vowel:
					count_vowel = count_vowel + 1
		if count_word == 0:
			return vowel_ratio
		else:
			vowel_ratio = float(count_vowel) / count_word
			return vowel_ratio
		
	def ReturnLable(self):
		if self.lable == 'dga':
			return 1
		else:
			return 0
		
	def ReturnData(self):
		return [self.GetLen(), self.GetEncropy(), self.GetVowel()]

def DataClean(filename):
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith("#") or line =="":
				continue
			item = line.split(',')
			domain_name = item[0]
			domain_lable = item[1]
			Domainlist.append(DomainClass(domain_name, domain_lable))
	f.close()

def test(filename):
	namelist = {}
	with open(filename) as f:
		for line in f:
			line = line.strip()
			if line.startswith("#") or line == "":
				continue
			domain = line
			namelist[domain] = DomainClass(domain, None).ReturnData()
	return namelist

def main():
	DataClean('train.txt')
	x_train, y_train = [], []
	for url in Domainlist:
		x_train.append(url.ReturnData())
		y_train.append(url.ReturnLable())
	clf = RandomForestClassifier(random_state=1)
	clf.fit(x_train,y_train)
	#print("the training process has finished")
	name = test('test.txt')
	result = "result.txt"
	f = open(result,'w')
	for item in name.keys():
		if clf.predict([name[item]]) == 1:
			f.write(item + "," + "dga\n")
		else:
			f.write(item + "," + "notdga\n")
	f.close()


if __name__ == '__main__':
	main()
