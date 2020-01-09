import os
import shutil

from glob import glob
from os.path import join as pj
from PIL import Image

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

def size_filter(classify_folder_path):
	"""
		Args: 
			classify_folder_path: the directory that contains the folder of each class

	"""
	# TODO: Add warning for this operation
	# Warning: Backup your data b4 using this
	img_path_list = []
	for cat in os.listdir(classify_folder_path):
		img_path_list += glob(pj(classify_folder_path, cat, '*'))
	for img_path in img_path_list:
		img = Image.open(img_path)
		shorter_side = min(img.size[0], img.size[1])
		if shorter_side < 25:
			os.remove(img_path)

if __name__ == '__main__':
	# for cat in ['ph', 'pl', 'pm']:
	# 	merge_cls('/media/yingges/Data_Junior/data/crop', cat)
	# set_split('/media/yingges/Data_Junior/data/classification/crop', '/media/yingges/Data_Junior/data/classification/20200107/test')
	# size_filter('/media/yingges/Data_Junior/data/classification/20200107_test_size_thres/val')

	size_filter('/media/yingges/Data_Junior/data/classification/20200107_all_size_thres/train')
	size_filter('/media/yingges/Data_Junior/data/classification/20200107_all_size_thres/val')