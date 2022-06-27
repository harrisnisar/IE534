import pandas as pd # load captions csv
import numpy as np # numpy used for mean
import torch # deep learning
import torchvision
from torch.utils.data import DataLoader 
import torchvision.transforms as transforms # transforms to prepare images
from nltk.translate.bleu_score import sentence_bleu
from main2 import Vocabulary, CaptionGenerator,  MyCollate, Flickr8kDataset

# this function follows the same path as the forward function in our caption 
# generation except returns the tokenized english caption 
 
def caption_image(model, device, image, vocabulary, max_length=50):
    result_caption = []
    
    model.eval()
    with torch.no_grad():
        image.to(device)
        x = model.encoder_model(image)
        states = None

        for _ in range(max_length):
            hiddens, states = model.decoder_model.rnn(x, states)
            output =  model.decoder_model.linear(hiddens.squeeze(0))
            predicted = output.argmax(1)
            result_caption.append(predicted.item())
            x =  model.decoder_model.embedder(predicted).unsqueeze(0)

            if vocabulary.index_to_string[predicted.item()] == "<EOS>":
                break

    return [vocabulary.index_to_string[cap_num] for cap_num in result_caption]


# generates a bleu scores for each image in dataloader, using the 5 included
# captions as our references
def bleu_scores(model, device,  dataloader,vocab, weights = (1,0,0,0)):
    scores = []
    refs = []
    for idx, (imgs, captions) in enumerate(dataloader):
        curr_ref = [vocab.index_to_string[cap_word.item()] for cap_word in captions[0]]
        refs.append(curr_ref[1:-1])
        if (idx+1)%5==0:
            candidate = caption_image(model, device, imgs, vocab)[1:-1]
            scores.append(sentence_bleu(refs, candidate, weights=weights))
            refs = []
    return(scores)
        
if __name__ == '__main__':
    
    #load paths, the data, and paramaters 
    models_directory = r'C:\Users\reuven\Documents\Classes\SavedModels\finalRun'
    image_directory = r'C:\Users\reuven\Documents\Classes\flicker8k\images'
    
    train_captions_path = r'C:\Users\reuven\Documents\Classes\flicker8kProcessed\flickr8k_train.txt'
    test_captions_path = r'C:\Users\reuven\Documents\Classes\flicker8kProcessed\flickr8k_test.txt'
    val_captions_path = r'C:\Users\reuven\Documents\Classes\flicker8kProcessed\flickr8k_val.txt'
    
    train_annotations_df = pd.read_csv (train_captions_path)
    val_annotations_df = pd.read_csv(val_captions_path)
    test_annotations_df = pd.read_csv (test_captions_path)
    
    
    train_caps = train_annotations_df['caption']
    val_caps = val_annotations_df['caption']
    test_caps = test_annotations_df['caption']
    
    combined_caps = pd.concat([train_caps, val_caps], axis = 0)
    combined_vocab = Vocabulary(5, combined_caps)
    test_vocab = Vocabulary(1,test_caps)
    
    image_transforms = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225],
            ),
        ])
    
    batch_size = 1
    emb_size = 512
    hidden_size = 1024
    test_dataset = Flickr8kDataset(image_directory, test_annotations_df, combined_vocab, transform=image_transforms)
    test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=MyCollate,
            drop_last=False
        )
        
    #set device to speed computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()
        print("Training on: ", torch.cuda.get_device_name(device_idx))
    else:
        print("Training on some CPU")
    
        
    
    bleu1_score_by_epoch = []
    bleu2_score_by_epoch = []
    bleu3_score_by_epoch = []
    bleu4_score_by_epoch = []
    
    for epoch in range(60):
        #load the model
        model_path = models_directory + '/model_params_epoch_no_' + str(epoch) + '.pt'            
        model = CaptionGenerator(len(combined_vocab), emb_size, hidden_size)
        model_save_state = torch.load(model_path, map_location=device)
        model.load_state_dict(model_save_state['model_state_dict'])
        
        #calculate each of the bleu scores 
        bleu1_scores = bleu_scores(model, device, test_loader, combined_vocab)
        bleu2_scores = bleu_scores(model, device, test_loader, combined_vocab, 
                                      weights = (.5, .5, 0, 0))
        bleu3_scores = bleu_scores(model, device, test_loader, combined_vocab, 
                                      weights = (.33, .33, .33, 0))
        bleu4_scores = bleu_scores(model, device, test_loader, combined_vocab, 
                                      weights = (.25, .25, .25, .25))
        
        bleu1_score_by_epoch.append(np.mean(bleu1_scores))
        bleu2_score_by_epoch.append(np.mean(bleu2_scores))
        bleu3_score_by_epoch.append(np.mean(bleu3_scores))
        bleu4_score_by_epoch.append(np.mean(bleu4_scores))

#combine and save scores
bleu_scores_by_epoch = pd.DataFrame()
bleu_scores_by_epoch["bleu1"] = bleu1_score_by_epoch
bleu_scores_by_epoch["bleu2"] = bleu2_score_by_epoch
bleu_scores_by_epoch["bleu3"] = bleu3_score_by_epoch
bleu_scores_by_epoch["bleu4"] = bleu4_score_by_epoch

bleu_scores_by_epoch.to_csv("bleu_scores_by_epoch_cont.csv")
    
    
    

#    print('Mean Bleu1 Score: ', np.mean(bleu1_scores))
#    print('Mean Bleu2 Score: ', np.mean(bleu2_scores))
#    print('Mean Bleu3 Score: ', np.mean(bleu3_scores))
#    print('Mean Bleu4 Score: ', np.mean(bleu4_scores))


#for idx, (imgs, captions) in enumerate(test_loader):
#    print(caption_image(imgs, combined_vocab))
#    print([combined_vocab.index_to_string[idx.item()] for idx in captions[0]])
#    break
    
    
