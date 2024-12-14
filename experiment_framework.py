"""
The goals for this file is to provide a framework for experimenting with different
training techniques to achieve good (or even better) results for Romanian Semantic Textual Similarity
"""


from datasets import load_dataset

ro_sts = load_dataset("ro-sts")
train_split = ro_sts.get_data("test")
print(train_split.head())

biblical_01 = load_dataset("biblical_01")
train_split = biblical_01.get_data("test")
print(train_split.head())