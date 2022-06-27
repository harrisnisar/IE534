import json
import numpy as np
import pandas as pd
import os

if __name__ == '__main__':
    
    # file paths, CHANGE THESE TO WHERE EVER YOU SAVED DATA
    data_directory = r'/u/eot/birnbam2/scratch/processed'
    
    
    splits = ['train', 'val', 'test']
    
    split_file = open(os.path.join(data_directory, 'dataset_flickr8k.json'))
  
    # returns JSON object as 
    # a dictionary
    data = json.load(split_file)
    samples = data['images']
    
    for split in splits:
        
        file_names = []
        captions = []
        
        for sample in samples:
            
            split_type = sample['split']
            file_name = sample['filename']
            sentences = sample['sentences']
            
            for sentence in sentences:
                raw_sentence = sentence['raw']
                
            
                if(split_type == split):
                    file_names.append(file_name)
                    captions.append(raw_sentence)

        df = pd.DataFrame(np.column_stack([file_names, captions,]), 
                               columns=['image', 'caption'])
        file_name = 'flickr8k_' + split + '.txt'
        save_path = os.path.join(data_directory, file_name)
        df.to_csv(save_path, header=['image', 'caption'], index=None, sep=',')