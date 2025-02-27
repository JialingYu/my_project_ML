# Supervised Fine-tune the GPT2 Model to be a Text Spam Classifier

## Steps:

- download a dataset contains text messages with label 'spam' and 'ham'(not spam').
- balance the dataset; sample equal number of spam and ham texts from the dataset
- split the dataset to train, validate, test dataset
- organize the datasets into a pytorch Dataset class instances
  1. use the GPT2 tokenizer to encode the texts to lists of integers encoded_ids
  2. truncate and pad the encoded_ids with 50256 (the id for <|endoftext|> token of GPT2 tokenizer)such that they are of the same length and are within the token context size(1024)of GPT2
  3. take the encode_ids to be features of the Dataset
  4. map the labels 'spam', 'ham' to 1 and 0 and take them to be labels of the Dataset
- organize the Dataset class instances into pytorch DataLoader instances
  - collate the data to batches of size 8
- instantiate a GPT2 model architecture written before
- download the weights of the pretrained GPT2 model of openai from transformers
- load the weights to the model architecture
- replace the last linear layer `nn.Linear(768,50257)` of the gpt2 model with a classification head `nn.Linear(768,2)` with weights default to be trainable
- set the `requires_grad` atrribute of the weights of the last transformer block, the final layer normalization to be True(make the weights trainable)
- use the above dataloaders to fine-tune the model
  - pass teh features of teh dataloader to the model
  - the model output is of shape (batch_size,seq_len,2)
  - take the last dimension of the seq_len to be the output logits: (batch_size,seq_len,2)[:,-1,:] since the last token of the text message has seen all the previous tokens and thus has encoded all the infos of the text message.
- evaluate the model loss and classification accuracy on the validation and test dataset seperated from the original dataset
- define new text messages and use the fine-tuned model to classify newly defined texts

## Conclusion:

The final model has high accuracy in the test set seperated from the original dataset used for fine-tuning. But on the dataset we define ourselves, the classification result is satisfactory to some extent but not robust; i.e., even slightly modify the contents of the text message will change the classification result.
  

