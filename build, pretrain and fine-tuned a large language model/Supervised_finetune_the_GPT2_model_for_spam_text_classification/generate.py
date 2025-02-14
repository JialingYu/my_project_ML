import tiktoken
import torch
from gpt_model import ModelArgs, Mygpt2model

def text_encode(text: str,tokenizer):
    '''
    a function which use tokenizer to encode the text into token ids and add a batch dimension
    :param text: the text string to encode
    :param tokenizer:
    :return: encoded_tensors: shape: [batch_size=1,seq_len]
    '''
    encoded_tokens=tokenizer.encode(text,allowed_special={'<|endoftext|>'})
    #convert the list to tensor and use .unsqueeze(0) to add a batch dimension in the first dimension
    encoded_tensors=torch.tensor(encoded_tokens).unsqueeze(0)
    return encoded_tensors

def tokenids_decode(encoded_tokens:torch.Tensor,tokenizer):
    '''
    a function to decode the encoded tokens to text using the tokenizer
    :param encoded_tokens: input tensor of shape (batch_size=1,n_tokens)
    :param tokenizer:
    :return: the decoded text string
    '''
    #.squeeze(0) remove the first dimension if it is of size 1
    encoded_tokens=encoded_tokens.squeeze(0)
    #encoded_tokens shape: [n_tokens,]
    #convert the tensor to list
    encoded_tokens=encoded_tokens.tolist()
    return tokenizer.decode(encoded_tokens)

#define a function to pass text to model and generate outputs
def generate_tokens_greedy(encoded_tokens:torch.Tensor,model,n_generate:int,model_context_len:int):
    '''
    a function which input the encoded tokens tensor in shape (batch_size,n_tokens),
    pass it to the model n_generate time to generate one token at a time. At each timestep,
    use greedy decoding to choose the token with the highest probability as the next token.
    Append the token to the input text and pass them to the model again.
    :param encoded_tokens: tensor in shape (batch_size,n_tokens)
    :param model: the model used for generation
    :param n_generate: number of tokens to generate
    :param model_context_len: the size of the context window of the model
    :return: encoded_tokens: the generated tokens with the original tokens of shape (batch_size,n_tokens_n_generate)
    '''
    model.eval()
    for _ in range(n_generate):
        #take the last part of the tokens if it is out of the range of the context length of the model
        encoded_tokens=encoded_tokens[:,-model_context_len:]
        #use context manager to close the computation of gradient and save memory
        with torch.no_grad():
            logits=model(encoded_tokens) #logits shape:(batch_size, n_tokens, 50257)
        next_tokens=logits[:,-1,:] # (batch_size,50257) the newly generated token is the last token
        probs=torch.softmax(next_tokens,dim=-1) #apply softmax to get the probabilities
        next_ids=torch.argmax(probs,dim=-1,keepdim=True) #(batch_size,1)
        encoded_tokens=torch.cat((encoded_tokens,next_ids),dim=-1)#(batch_size,n_tokens+1)
    return encoded_tokens



#define a function to generate text after start_text after epoch during training
def generate_and_print_sample(model,conj,start_text:str,tokenizer,device,n_generate:int):
    '''
    tokenize start_text using tokenizer, move it to device,
    and pass it to model to generate n_generate new words
    :param model: the model to generate output
    :param start_text: the text inout to model
    :param tokenizer: the tokenizer for the model
    :param device: the device to use
    :param n_generate: the number of tokens to generated
    :return:None
    '''
    model.to(device)
    model.eval()
    #tokenizer the start_text into tokens and move it to device
    encoded_tokens=text_encode(start_text,tokenizer).to(device)
    #encoded_tokens shape (batch_size=1, n_tokens)
    #get the context window size of the model
    model_context_len=conj['context_length']
    #pass the token_ids to model to generate output
    with torch.no_grad():
        out_tokens=generate_tokens_greedy(encoded_tokens,model,n_generate,model_context_len)
    #out_tokens shape (batch_size=1, n_tokens)
    #decode the output
    out_text=tokenids_decode(out_tokens,tokenizer)
    print(f'\nThe output text of the current epoch is:\n{out_text}.')
    #convert back the model to training mode
    #model.train()

def top_k_and_temperature(encoded_tokens,model,n_generate,model_context_len,device,top_k=None,
                          temperature=0):
    '''use the model for inference, input start_text to the model and get the output logits,
    apply top_k sampling to only choose the top_k highest tokens as condidates for next token,
    apply softmax to get the probabilities for the k candidates,
    scale the logits by temperature and sample from the scaled'''
    model.to(device)
    model.eval()
    for _ in range(n_generate):
        #only take the last part of encoded_ids within context len
        encoded_tokens=encoded_tokens[:,-model_context_len:]
        #print(encoded_tokens)
        with torch.no_grad():
            logits=model(encoded_tokens)
            #logits shape: (batch_size, n_tokens,vocab_size)
        #get the next tokens
        #next_tokens shape:(batch_size,vocab_size)
        next_tokens=logits[:,-1,:]
        if top_k is not None:
            values,indices=torch.topk(next_tokens,top_k)
            #values shape: (batch_size,top_k)
            #print(values.shape,indices.shape)
            #take the smallest value from the top k
            values_min=values[:,-1]
            #print('values_min:', values_min)
            #values_min shape: (batch_size,)
            #print(values_min.shape)
            #if the condition is satisfied, take values from input
            #otherwise, take input from other
            #mask the non top k values into -inf
            next_tokens=torch.where(next_tokens<values_min,
                                    torch.tensor(float('-inf')).to(logits.device),
                                    next_tokens)
            #print('next_token shape:',next_tokens)
            #print('next_tokens:',(next_tokens!= torch.tensor(float('-inf'))).sum(dim=-1))
        #if we use temperature scaling, we divide the next_tokens by the temperature
        #and apply softamex to get the probability and use multinomial to sample from
        #the probability, otherwise we use argmax to get the topen with max prob directly
        if temperature>0:
            next_tokens=next_tokens/temperature
            probs=torch.softmax(next_tokens,dim=-1)
            #for each row of probs, sample one sample from its probability distribution
            next_ids=torch.multinomial(probs,1)
            #print('next_ids',next_ids)
        else:
            next_ids=torch.argmax(next_tokens,dim=-1,keepdim=True)
        encoded_tokens=torch.cat((encoded_tokens,next_ids),dim=-1)
    return encoded_tokens

if __name__ == '__main__':
    # tokenize a sentence to try the model
    s = 'Today is Friday,'
    # use the gpt2 tokenizer
    enc = tiktoken.get_encoding('gpt2')
    encoded_tensors = text_encode(s, enc)
    print(f'The input tokens is of size {encoded_tensors.shape}.')
    args=ModelArgs()
    model=Mygpt2model(args)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generate_and_print_sample(model, s, enc, device, 20)