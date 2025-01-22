# Build and pretrain a gpt2 model

In the notebook, we build a gpt2 model and pretrain it using a small data set.

These consists of the following steps:

1. build the gpt2 architecture which consists of 12 attention layers.
2. build the dataset and data loader for pretraining.
3. pretrain the gpt2 model using the dataloader for 10 epochs; compute the training and validation loss
4. plot the training and validation loss for model evaluation
5. use the pretrained model for inference
6. use top-k sampling and temperature scaling to improve text generation.
7. save the pretrained model and optimizer for future use.
