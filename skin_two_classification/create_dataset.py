""" 
@ author: Qmh
@ file_name: create_dataset.py
@ time: 2019:11:15:20:39
""" 
import json
import os
import tqdm
from PIL import Image
from PIL import ImageFile
from collections import Counter

ImageFile.LOAD_TRUNCATED_IMAGES = True

DATA_DIR = '/home/dsp/Documents/skin_disease/data/dataset'

def tag_dataset_pictures(json_path):
    '''
    统计数据集的标签
    '''
    with open(json_path,'rb') as f:
        dict1 = json.load(f)
        wrong_files = []
        labels = []
        for key in tqdm.tqdm(dict1.keys()):
            txt_name = key + '.txt'
            sample = dict1[key]
            clinical = sample['meta']['clinical']
            if 'benign_malignant' not in clinical.keys():
                continue
            if clinical['benign_malignant'] in ["",None]:
                wrong_files.append(key)
                continue
            else:
                with open(os.path.join(DATA_DIR,txt_name),'w') as f1:
                    labels.append(clinical['benign_malignant'])
                    f1.write(clinical['benign_malignant'])
    print(wrong_files)
    print(Counter(labels))


def create_maliganant_set(json_path,txt_path):
    '''
    筛选出患病的图片路径
    '''
    # 11405
    with open(json_path,'rb') as f:
        dict1 = json.load(f)
        wrong_files = []
        mal_img_names,mal_labels = [],[]
        benign_img_names,benign_labels = [],[]
        for key in tqdm.tqdm(dict1.keys()):
            txt_name = key + '.txt'
            sample = dict1[key]
            clinical = sample['meta']['clinical']
            if 'benign_malignant' not in clinical.keys():
                continue
            elif clinical['benign_malignant'] == 'malignant':
                mal_img_names.append(key+".jpg")
                mal_labels.append(clinical['benign_malignant'])
            elif clinical['benign_malignant'] == 'benign':
                benign_img_names.append(key+'.jpg')
                benign_labels.append(clinical['benign_malignant'])
            else:
                continue
        # 写入txt
        benign_len = int(len(benign_labels)*0.2)
        mal_img_names.extend(benign_img_names[:benign_len])
        mal_labels.extend(benign_labels[:benign_len])
        data_set = (mal_img_names,mal_labels)
        write_dataset_to_txt(data_set,txt_path)

def write_dataset_to_txt(data_set,txt_path):
    '''
    将数据集的路径写入txt文件保存
    data_set: 保存图片路径和标签的元组
    txt_path： 待保存的txt文件路径
    '''
    img_paths,labels = data_set

    with open(txt_path,'w') as f:
        for index,img_path in enumerate(img_paths):
            f.write(img_path+","+str(labels[index])+'\n')
    print(f"write to {txt_path} successed")        


def save_maliganant_set(txt_path,dataset_dir,save_path):
    '''
    保存患病的图片
    '''
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(txt_path,'r') as f:
        img_names = f.read().split('\n')
    for img_name in tqdm.tqdm(img_names):
        img_name = img_name.split(',')[0]
        img_path = os.path.join(dataset_dir,img_name)
        img = Image.open(img_path)
        img.save(os.path.join(save_path,img_name))




if __name__ == "__main__":
    json_path = './dataset/metadata.json'
    # tag_dataset_pictures(json_path)
    txt_path = './dataset/small_dataset.txt'

    # create_maliganant_set(json_path,txt_path)

    save_path = '/home/dsp/Documents/skin_disease/used_dataset'
    dataset_dir = '/home/dsp/Documents/skin_disease/data/dataset'
    save_maliganant_set(txt_path,dataset_dir,save_path)

    