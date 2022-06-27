import pandas as pd # load captions csv
from PIL import Image # load images
import os # file path stuff

import numpy as np # numpy used for mean

import torch # deep learning
from torch import nn # networks
import torchvision # torchvision models
import torchvision.transforms as transforms # transforms to prepare images
from torch.nn.utils.rnn import pad_sequence  # pad batches
from torch.utils.data import Dataset # create a custom flickr8k dataset
from torch.utils.data import DataLoader # create a custom loader with custom collate function to pad batches
import torch.optim as optim # to optimize model
from My_Adam import My_Adam # some sort of adam that works better on the bluewaters python version?

# TO-DOs:
    # add training-testing split DONE
    # implement inference (naive, beam-search)
    # infer captions for the testing set
    # calculate bleu scores between infered captions and actual captions
    # save training and testing metrics (loss, bleu) and model DONE
    # run this on blue waters DONE
    
# datasets:
    # flickr8k: https://www.kaggle.com/dataset/e1cd22253a9b23b073794872bf565648ddbe4f17e7fa9e74766ad3707141adeb
    
# sources:
    # https://www.youtube.com/watch?v=y2BaTt1fxJU
    # https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/more_advanced/image_captioning

# vocab class helps us define mappings from word to index and index to word based on some freq threshoold
# also has function to convert a string of text into numerical representation based on mappings
# ex: "I am Harris Nisar and I am coding" -> [1,3,4,5,6,7,3,4,8,2]
class Vocabulary:
    def __init__(self, freq_threshold, sentence_list):
        self.index_to_string = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.string_to_index = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
        self.freq_threshold = freq_threshold
        self.build_vocab(sentence_list)
            
    def english_tokenizer(self, text):
        return [tok.lower() for tok in text.split()]
    
    def __len__(self):
        return len(self.string_to_index)
    
    def build_vocab(self, sentence_list):
        frequencies = {}
        idx = 4
        for sentence in sentence_list:
            tokenized_sentence = self.english_tokenizer(sentence)
            for word in tokenized_sentence:
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.string_to_index[word] = idx
                    self.index_to_string[idx] = word
                    idx += 1
    
    def numericalize(self, text):
        return [
            self.string_to_index[token] if token in self.string_to_index else self.string_to_index["<UNK>"]
            for token in self.english_tokenizer(text)
        ]
        
# Flickr8k dataset that returns an image and caption for a particular idx in the __getitem__ method
class Flickr8kDataset(Dataset):
    def __init__(self, image_directory, annotation_df, vocabulary, transform=None):
        self.annotations_df = annotation_df
        # ~ 8k images
        # 5 caps per image
        # ~ 40k samples
        self.image_directory = image_directory  
        self.image_size = 224
        self.transform = transform
        
        self.image_paths = self.annotations_df['image']
        self.captions = self.annotations_df['caption']
        
        self.vocabulary = vocabulary

    def __len__(self):
        return self.captions.size

    def __getitem__(self, idx):
        image_file_name = self.image_paths[idx]
        image_path = os.path.join(self.image_directory, image_file_name)
        image = Image.open(image_path).resize((self.image_size,self.image_size), 0)
        
        if self.transform:
            image = self.transform(image)

        caption = self.captions[idx]
        numericalized_caption = [self.vocabulary.string_to_index["<SOS>"]] # [1]
        numericalized_caption += self.vocabulary.numericalize(caption) # [1 12 33 455 ... 322]
        numericalized_caption.append(self.vocabulary.string_to_index["<EOS>"]) # [1 12 33 455 ... 322 2]
        return image, torch.tensor(numericalized_caption) # (3,256,256), (1,cap_len+2)

# custom collate function that runs when we are batching to allow us to pad batches
def MyCollate(batch):
    imgs = [item[0].unsqueeze(0) for item in batch]
    imgs = torch.cat(imgs, dim=0)
    targets = [item[1] for item in batch]
    targets = pad_sequence(targets, batch_first=True, padding_value=0)

    return imgs, targets # (batch_size, 3, 256, 256), (batch_size, max_batch_cap_length + 2)

# encoder takes in images and maps them into the same space as decoder output
# by passing it through some pretrained conv-net with a fully connected layer
# that does the mapping
# output: batch_size, 1, embedding_size
class Encoder(nn.Module):
    def __init__(self, emb_size=512):
        super(Encoder, self).__init__()
        self.emb_size = emb_size
        self.conv_net = torchvision.models.resnet101(pretrained=True) 
        children_resnet_ft = list(self.conv_net.children())
        for layer in children_resnet_ft:
            for p in layer.parameters():
                p.requires_grad = False
            
        self.conv_net.fc = nn.Linear(self.conv_net.fc.in_features, emb_size)
        for param in self.conv_net.fc.parameters():
            param.requires_grad = True
        
        self.dropout = nn.Dropout(p=0.5)
        # for name, param in self.conv_net.named_parameters():
        #     print(name, param.requires_grad)
            
    def forward(self, images):
        encoded_images = self.conv_net(images)  
        encoded_images = encoded_images.view(encoded_images.shape[0], -1)
        encoded_images = encoded_images.unsqueeze(1)
        return self.dropout(encoded_images)

# decoder takes in encoded images and captions, it encodes captions, concats 
# it with images and passes that into the RNN of choice
# max_batch_cap_length includes start, stop, and pad
# images (batch_size, 1, 512), tokenized_caps (batch_size, max_batch_cap_length)
# embed_captions, (batch_size, max_batch_cap_length, 512)
# rnn input, (batch_size, max_batch_cap_length + 1, 512)
# hidden_state, (batch_size, max_batch_cap_length + 1, hidden_size)
# output (batch_size, max_batch_cap_length + 1, vocab_size)
class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size):
        super(Decoder, self).__init__()
        self.embedder = nn.Embedding(vocab_size, emb_size)
        self.rnn = nn.LSTM(emb_size, hidden_size, 1, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, encoded_images, tokenized_caps):
        embeded_caps = self.embedder(tokenized_caps)
        rnn_input = torch.cat((encoded_images, embeded_caps), 1)
        hidden_state = self.rnn(rnn_input)[0]
        return self.dropout(self.linear(hidden_state))
  
# glue model class that combines our encoder and decoder
class CaptionGenerator(nn.Module):
    def __init__(self, vocab_size, emb_size, hidden_size):
        super(CaptionGenerator, self).__init__()
        self.encoder_model = Encoder(emb_size=512)
        self.decoder_model = Decoder(vocab_size, emb_size, hidden_size)
        
    def forward(self, images, captions):
        encoder_output = self.encoder_model(images)
        decoder_output = self.decoder_model(encoder_output, captions[:,:-1]) # look into this a bit more 
        return decoder_output

# train from scratch
def train(train_dataset, val_dataset, train_loader, val_loader, model, criterion, optimizer, scheduler,
          save_dir=None, initial_batch_number=0, device="cpu", num_epochs=100):
     # printing/saving stuff
    print_every = 1
    save_every = 1
    num_training_batches = len(train_dataset)/batch_size
    num_val_batches = len(val_dataset)/batch_size
    num_val_batches = len(val_dataset)/batch_size
  
    training_epoch_loss = []
    val_epoch_loss = []
    model.to(device)
    # training/validation loop
    for epoch in range(num_epochs):        
        print('EPOCH # : ', epoch)
        print('-'*20)
        # where to save the model every epoch
        save_path = save_dir + 'moedel_params_epoch_no_' + str(epoch) + '.pt'
        
        # training/validation batch losses
        training_batch_loss = []
        val_batch_loss = []
        
        # current learning rate (will decay per epoch)
        current_lr = optimizer.param_groups[0]['lr']
        # training  
        model.train()
        for idx, (imgs, captions) in enumerate(train_loader):
            # go thru 2 batches for testing the code
            if idx == 2:
                break
            
            # move data to gpu/cpu
            imgs = imgs.to(device)
            captions = captions.to(device) # (batch_size, max_cap_length)
            
            # what has the scheulder set our learning rate to
            
            
            # pass data thru our model
            train_model_output = model(imgs, captions) # (batch_size, max_cap_length, vocab_size) loss function expects (N, vocab_size)
            
            # calculate loss between model output and labels (captions)
            train_loss = criterion(train_model_output.reshape(-1, train_model_output.shape[2]), captions.reshape(-1))
            training_batch_loss.append(train_loss.item())
            
            # optimze
            optimizer.zero_grad()
            train_loss.backward(train_loss)
            optimizer.step()
           
            current_batch_number = idx+initial_batch_number
            
            # printing stuff
            # if(idx%print_every==0):
            #     print('epoch # ', epoch, '/' ,num_epochs,
            #           'batch ', current_batch_number+1, '/', num_training_batches,
            #           'lr: ', current_lr,
            #           'train loss: ', train_loss.item())
                
        # validation    
        model.eval()
        with torch.no_grad():
            for idx, (imgs, captions) in enumerate(val_loader):
                # go thru 2 batches for testing the code
                if idx == 2:
                    break
                
                # move data to gpu/cpu
                imgs = imgs.to(device)
                captions = captions.to(device) # (batch_size, max_cap_length)
                
                # pass data thru our model
                val_model_output = model(imgs, captions) # (batch_size, max_cap_length, vocab_size) loss function expects (N, vocab_size)
                
                # calculate loss between model output and labels (captions)
                val_loss = criterion(val_model_output.reshape(-1, val_model_output.shape[2]), captions.reshape(-1))
                val_batch_loss.append(val_loss.item())
                
                current_batch_number = idx+initial_batch_number
                
                # # printing stuff
                # if(idx%print_every==0):
                #     print('epoch # ', epoch, '/' ,num_epochs,
                #           'batch ', current_batch_number+1, '/', num_val_batches,
                #           'train loss: ', 
                #           'val loss: ', val_loss.item())
        
        # decay lr
        scheduler.step()
        
        # mean epoch losses
        mean_epoch_train_loss = np.mean(training_batch_loss)
        mean_epoch_val_loss = np.mean(val_batch_loss)
        training_epoch_loss.append(mean_epoch_train_loss)
        val_epoch_loss.append(mean_epoch_val_loss)
        
        # printing per epoch
        print('Completed epoch # ', epoch, '/' ,num_epochs,
                'mean train loss: ', mean_epoch_train_loss,
              'mean val loss: ',mean_epoch_val_loss)
        
        # saving every save_everyth epoch
        if(epoch%save_every==0 and save_dir!=None):
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss':mean_epoch_train_loss,
                'val_loss':mean_epoch_val_loss
                }, save_path)
            
        # printing some seperating lines    
        print('-'*20)
        print('-'*20)

    
if __name__ == '__main__':
    # if this is true, we are training locally, if not we are using Bluewaters 
    is_training_local = True
    
    if(is_training_local):
        # file paths, CHANGE THESE TO WHERE EVER YOU SAVED DATA
        ## harris local directories
        image_directory = r'F:\Productivity\School\flickr8k\images'
        train_captions_path = r'F:\Productivity\School\flickr8k\flickr8k_train.txt'
        val_captions_path = r'F:\Productivity\School\flickr8k\flickr8k_val.txt'
        save_dir = r'F:/Productivity/School/trained_models/show_and_tell/' 
    else: 
        ## bluewater directories
        image_directory = '/u/eot/birnbam2/scratch/flickr8k/images'
        train_captions_path = '/u/eot/birnbam2/scratch/processed/flickr8k_train.txt'
        val_captions_path = '/u/eot/birnbam2/scratch/processed/flickr8k_val.txt'
        save_dir = '/u/eot/birnbam2/scratch/SavedModels/'
    
    ## GENERATING VOCAB FROM ALL TRAINING AND VALIDATION CAPTIONS
    # load train and val captions
    train_annotations_df = pd.read_csv (train_captions_path)
    val_annotations_df = pd.read_csv(val_captions_path)
    train_caps = train_annotations_df['caption']
    val_caps = val_annotations_df['caption']
    
    # combine them
    combined_caps = pd.concat([train_caps, val_caps], axis = 0)
    
    # create a Vocabulary based on this combined dataset
    combined_vocab = Vocabulary(5, combined_caps)
    
    # gpu or cpu    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device_idx = torch.cuda.current_device()
        print("Training on: ", torch.cuda.get_device_name(device_idx))
    else:
        print("Training on some CPU")
    
    # dataset hyperparams
    batch_size = 32
    
    # model hyperparams
    emb_size = 512
    hidden_size = 1200 
    
    # optimizer/scheduler hyperparams
    lr = 0.0005
    step_size =30
    gamma=0.1
    
    # transforms for our images
    image_transforms = transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225],
            ),
        ])
    
    ## flickr8k dataset and data loaders
    # train
    train_dataset = Flickr8kDataset(image_directory, train_annotations_df, combined_vocab, transform=image_transforms)
    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=MyCollate,
            drop_last=True
        )
    
    # val
    val_dataset = Flickr8kDataset(image_directory, val_annotations_df, combined_vocab, transform=image_transforms)
    val_loader = DataLoader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=MyCollate,
            drop_last=True
        )
    
    # caption paramaters
    pad_idx = 0
    
    # loss function
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    # model
    model = CaptionGenerator(len(combined_vocab), emb_size, hidden_size)
    model.to(device)
    
    # optimizer/scheuler
    optimizer = My_Adam(model.parameters(), lr = lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # model/optimizer stuff goes here:
    # if(checkpoint_path!=None):
    #     loaded_model = torch.load(checkpoint_path)
    #     model.load_state_dict(loaded_model['model_state_dict'])
    #     optimizer.load_state_dict(loaded_model['optimizer_state_dict'])
    #     epoch = loaded_model['epoch']
    #     current_loss_value = loaded_model['loss']
    #     initial_batch_number = loaded_model['batch_number']
        
    # train from scratch
    train(train_dataset, val_dataset, 
          train_loader, val_loader, 
          model, criterion, 
          optimizer, scheduler,
          save_dir = save_dir, device=device, num_epochs=100)