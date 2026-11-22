import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
eval_iters = 200
learning_rate = 1e-3
max_iters = 5000
eval_interval = 300
n_embd = 32

torch.manual_seed(1337)

with open('/Users/manasa.mylavarapu/Desktop/AK_Practice/Transform_her/input.txt', 'r') as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars) #65

stoi = {ch : i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[ch] for ch in s]

def decode(ids):
    return ''.join(itos[i] for i in ids)

data = torch.tensor(encode(text), dtype = torch.long)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

x = train_data[:block_size]
y = train_data[1: block_size+1]

def get_batch(split):
    data_ = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_)- block_size, (batch_size,))
    x_ = torch.stack([data_[i: i+block_size] for i in ix])
    y_ = torch.stack([data_[i+1: i+1+block_size] for i in ix])   
    return x_, y_

@torch.no_grad() #tells torch -> do not build a computational graph for tracking the gradients
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    "one head of self attention"
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # (B,T,C)
        q = self.query(x)
        # attention scores
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) -> (B, T, T)
        wei = wei.masked_fill((self.tril[:T,:T])== 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)
        
        v = self.value(x)
        out = wei @ v #(B,T,T) @ (B, T,C) -> (B,T,C)
        return out
    
class MultiHeadAttention(nn.Module):
    "multiple heads of self attention in parallel"
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    
    def forward(self, x):
        return torch.cat([h(x) for h in self.heads], dim = -1)

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_heads = MultiHeadAttention(4, n_embd//4)
        self.lm_head = nn.Linear(n_embd , vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) #(B, T, n_embd) 
        #m(xb) └──> nn.Module.__call__(self, xb, yb)└──> BigramLanguageModel.forward(self, idx=xb)
        pos_emb = self.position_embedding_table(torch.arange(T))
        x = tok_emb + pos_emb
        x = self.sa_heads(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        if targets is None:
            loss = None  
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond) # logits (B, T,C)
            probs = F.softmax(logits[:, -1, :], dim = -1) #probs (B,C)
            idx_next = torch.multinomial(probs, num_samples=1) 
        
            idx = torch.cat((idx, idx_next), dim = 1)
            
        return idx
    
model = BigramLanguageModel(vocab_size)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss = {losses['train']:.4f}, val loss = {losses['val']:.4f}")
        
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none = True)
    loss.backward() # computes gradients
    optimizer.step() # updates the weights
    
context = torch.zeros((1, 1), dtype=torch.long)
print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))