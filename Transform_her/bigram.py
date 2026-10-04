import torch
import torch.nn as nn


with open('/Users/manasa.mylavarapu/Desktop/AK_Practice/Transform_her/input.txt', 'r') as f:
    text = f.read()

chars = sorted(set(text))

vocab_size = len(chars)

stoi = {ch : i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

def encode(s):
    return [stoi[ch] for ch in s]

def decode(ids):
    return ''.join(itos[i] for i in ids)

# print(vocab_size)
# print(''.join(chars))

# print(decode(encode("hii there baby")))

data = torch.tensor(encode(text), dtype = torch.long)
# print(data[:100])
# print(decode(data[:100].tolist()))

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]
# print(len(train_data))
# print(len(val_data))
block_size = 8
x = train_data[:block_size]
y = train_data[1: block_size+1]
for t in range(block_size):
    context = x[: t+1]
    target = y[t]
    # print(f"when input is {context.tolist()} the target is {target}")

batch_size = 4
def get_batch(split):
    data_ = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_)- block_size, (batch_size,))
    x_ = torch.stack([data_[i: i+block_size] for i in ix])
    y_ = torch.stack([data_[i+1: i+1+block_size] for i in ix])   
    return x_, y_
xb, yb = get_batch('train')
# print(xb.shape)
# print(xb)
# print(yb)

# def add_item(cart, name , price, qty = 1):
#     cart.append({"name" : name, "price" : price, "qty" : qty})
# def remove_item(cart, name):
#     cart = [i for i in cart if i["name"] != name]
    
# class Cart:
#     def __init__(self):
#         self._items = [] # state
        
#     def add(self, name, price, qty=1):
#         self._items.append({"name": name, "price" : price, "qty" : qty})
#     def remove(self, name):
#         self._items =  [i for i in self._items if i["name"] != name]
#     def total(self, discount = 0):
#         subtotal = sum(i["price"] * i["qty"] for i in self._items)
#         return subtotal * (1- discount)

# c = Cart()
# c.add('Milk', 30, 2)
# c.add('curd', 32, 1)
# c.add('bread', 40, 2)
# c.remove('curd')
# print(c.total(discount = 0))

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx):
        logits = self.token_embedding_table(idx)
        return logits