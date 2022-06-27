# CS 547 Group 3 Project

Our implementation of  [Show and Tell, A Neural Imaging Caption Generator](https://arxiv.org/abs/1411.4555)




## Usage

1. Download Captions and Images from [Flicker8k](https://www.kaggle.com/kunalgupta2616/flickr-8k-images-with-captions). 
2. Adapt split_loader.py for your file specifications (dataset_flickr8k.json location, caption/image locations, and output:train, validation, testing txt files)
3. Run to get  train, validation, testing partitons 
4. Adapt train_models.py for your file specifications (caption/image locations, train/validation partitions, output: models at specified times), and any hyperparamaters you would like
5. Run train_model.py (note we use a My_Adam.py instead of torch.optim.adam as there is a known value error which can occur in thee torch verison on the BlueWater  supercomputer)
5. Adapt infer_from_models.py for your file specifications (caption/image locations, test partitions, model locations, output: bleu scores for each saved model) 
6. Run infer_from_models.py
