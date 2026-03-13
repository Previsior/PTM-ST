import os
import re
import json
from PIL import Image
from torch.utils.data import Dataset
from functools import lru_cache

def pre_caption(caption, max_words=50):
    caption = re.sub(
        r"([.!\"()*#:;~])",       
        ' ',
        caption.lower(),
    )
    caption = re.sub(
        r"\s{2,}",
        ' ',
        caption,
    )
    caption = caption.rstrip('\n') 
    caption = caption.strip(' ')

    # truncate caption
    caption_words = caption.split(' ')
    if len(caption_words)>max_words:
        caption = ' '.join(caption_words[:max_words])
            
    return caption


class flickr30k_train(Dataset):
    def __init__(self, transform, image_root, ann_root, max_words=30, prompt='', json_path=None):        
        filename = 'flickr30k_train.json'
        self.annotation = json.load(open(os.path.join(ann_root, filename),'r'))
        self.transform = transform
        self.image_root = image_root
        self.max_words = max_words      
        self.prompt = prompt
        if json_path:
            with open(json_path, 'r', encoding='utf-8') as f:
                cluster_to_data = json.load(f)
            self.cluster_to_data = cluster_to_data
        
    def __len__(self):
        return len(self.annotation)
    
    @lru_cache(maxsize=100)
    def read_image(self, image_path):      
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)
        return image
    
    def __getitem__(self, index):    
        ann = self.annotation[index]
        image_path = os.path.join(self.image_root, ann['image'])
        image = self.read_image(image_path)
        
        caption = self.prompt + pre_caption(ann['caption'], self.max_words) 

        return image, caption
        
    def get_all_captions(self):
        captions = []
        for ann in self.annotation:
            caption = self.prompt + pre_caption(ann['caption'], self.max_words)
            captions.append(caption)
        return captions


class flickr30k_retrieval_eval(Dataset):
    def __init__(self, transform, image_root, ann_root, split, max_words=30):  
        '''
        image_root (string): Root directory of images (e.g. flickr30k/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        '''
        # urls = {'val':'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_val.json',
        #         'test':'https://storage.googleapis.com/sfr-vision-language-research/datasets/flickr30k_test.json'}
        filenames = {'val':'flickr30k_val.json','test':'flickr30k_test.json'}
        
        #######################
        # download_url(urls[split], ann_root)
        #######################
        
        self.annotation = json.load(open(os.path.join(ann_root,filenames[split]),'r'))
        self.transform  = transform
        self.image_root = image_root
        self.max_words = max_words   
        
        self.text = []
        self.image = []
        self.txt2img = {}
        self.img2txt = {}
        
        txt_id = 0
        for img_id, ann in enumerate(self.annotation):
            self.image.append(ann['image'])
            self.img2txt[img_id] = []
            for i, caption in enumerate(ann['caption']):
                self.text.append(pre_caption(caption,max_words))
                self.img2txt[img_id].append(txt_id)
                self.txt2img[txt_id] = img_id
                txt_id += 1
                                    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        image_path = os.path.join(self.image_root, self.annotation[index]['image'])       
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)

        return image, index
