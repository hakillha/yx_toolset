import os
import shutil

from glob import glob
from os.path import join as pj

from sklearn.model_selection import train_test_split

def merge_cls(data_path, target_cls):
	folder_list = [os.path.basename(path) for path in glob(pj(data_path, '*'))]
	folder_list = [path for path in folder_list if path.startswith(target_cls) and path != target_cls]
	file_list = []
	for folder in folder_list:
		file_list += glob(pj(data_path, folder, '*'))
	for file in file_list:
		shutil.move(file, pj(data_path, target_cls))

def set_split(data_path, testset_path, test_ratio=0.2):
	file_paths = []
	file_labels = []
	cat_list = [os.path.basename(path) for path in glob(pj(data_path, '*'))]
	for cat in cat_list:
		for file_path in glob(pj(data_path, cat, '*')):
			file_paths.append(file_path)
			file_labels.append(cat)

	test_size = int(len(file_paths) * test_ratio)
	# print(test_size)
	(trainPaths, testPaths, trainLabels, testLabels) = train_test_split(file_paths, 
																		file_labels, 
																		test_size=test_size, 
																		stratify=file_labels, 
																		random_state=42)
	for f_path, f_label in zip(testPaths, testLabels):
		target_dir = pj(testset_path, f_label)
		if not os.path.isdir(target_dir):
			os.mkdir(target_dir)
		shutil.move(f_path, target_dir)

if __name__ == '__main__':
	# for cat in ['ph', 'pl', 'pm']:
	# 	merge_cls('/media/yingges/Data_Junior/data/crop', cat)
	set_split('/media/yingges/Data_Junior/data/classification/crop', '/media/yingges/Data_Junior/data/classification/20200107/test')
