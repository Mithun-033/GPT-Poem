import re
import torch
import torch.nn as nn
from tokenizers import Tokenizer
import random

#Hyper Parameters
context_window_length=128
batch_size=256
n_embed=288
n_head=9
n_layers=8
v_size=9000
head_size=n_embed//n_head
temperature=0.63

device ="cuda:0" if torch.cuda.is_available() else "cpu"

class AttentionHead(nn.Module):
    def __init__(self,head_size):
        super().__init__()
        self.key=nn.Linear(n_embed,head_size,bias=False) #(B,T,C)-->(B,T,H)
        self.query=nn.Linear(n_embed,head_size,bias=False) #(B,T,C)-->(B,T,H)
        self.value=nn.Linear(n_embed,head_size,bias=False)  #(B,T,C)-->(B,T,H
        #self.dropout=nn.Dropout(0.2)

    def forward(self,x):
        k=self.key(x)     #(B,T,H)
        q=self.query(x)   #(B,T,H)
        v=self.value(x)   #(B,T,H)

        # Do Dot product of k and q

        weights=k@q.transpose(-2,-1)*head_size**-0.5  # (B,T,H) x (B,H,T) --> (B,T,T)
        T=x.size(1)
        mask=torch.tril(torch.ones(T,T,device=x.device))
        weights=weights.masked_fill(mask==0,float('-inf'))
        weights=nn.functional.softmax(weights,dim=-1)
        #weights = self.dropout(weights)

        output=weights@v #(B,T,T) x (B,T,H) --> (B,T,H)
        return output
    
class MultiHead(nn.Module):
    def __init__(self,n_head,head_size):
        super().__init__()
        self.heads=nn.ModuleList([AttentionHead(head_size) for _ in range(n_head)])
        #self.project=nn.Linear(n_head*head_size,n_embed)
        self.dropout=nn.Dropout(0.2)
    def forward(self,x):
        out=torch.cat([h(x) for h in self.heads],dim=-1)  # (B,T,H*N)
        #out=self.project(out)  # (B,T,H*N) --> (B,T,C) 
        out = self.dropout(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.FF=nn.Sequential(
            nn.Linear(n_embed,3*n_embed),
            nn.GELU(),
            nn.Linear(3*n_embed,n_embed),
            nn.Dropout(0.2)
        )

    def forward(self,x):
        return self.FF(x)
    
class Block(nn.Module):
    def __init__(self,n_embed,n_head):
        super().__init__()
        head_size=n_embed//n_head
        self.SelfAtt = MultiHead(n_head, head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self,x):
        x=x + self.SelfAtt(self.ln1(x)) 
        x=x + self.ffwd(self.ln2(x))
        return x  #(B,T,C)
    
class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed=nn.Embedding(v_size,n_embed)  # (B,T) --> (B,T,C)
        self.pos_embed=nn.Embedding(context_window_length,n_embed) # (T) --> (T,C)

        self.blocks=nn.Sequential(*[Block(n_embed,n_head) for _ in range(n_layers)])
        self.final_layernorm = nn.LayerNorm(n_embed) # final layer norm
        self.lm_head = nn.Linear(n_embed, v_size)

    def forward(self,x):
        # x ==> (B,T)

        tok_embeds=self.embed(x) # (B,T,C)
        pos_embeds=self.pos_embed(torch.arange(x.size(1),device=x.device)) #(T,C)
        x=tok_embeds + pos_embeds # pos_embed r broadcasted and added to every batch element

        x=self.blocks(x)
        x=self.final_layernorm(x)
        logits=self.lm_head(x)

        return logits


    @torch.no_grad()
    def generate(model,idx,max_new_tokens):
        for _ in range(max_new_tokens):
            if idx.size(1)>context_window_length:
                idx_cond=idx[:,-context_window_length:]
            else:
                idx_cond=idx

            logits=model(idx_cond)
            logits=logits[:,-1,:]
        
            logits[:,bad_token_ids]=-1e9
            logits=logits/temperature
            probs=torch.softmax(logits,dim=-1)
            next_token=torch.multinomial(probs,1)
            idx=torch.cat((idx,next_token),dim=1)

        return idx

    
tokenizer=Tokenizer.from_file("Tokenizer (1).json")
sd=torch.load("Model_FineTuned.pt",map_location=device)
sd={k.replace("_orig_mod.",""):v for k,v in sd.items()}
model=GPT()
model.load_state_dict(sd)
model.to(device)
model.eval()

bad_token_ids=[]

for i in range(tokenizer.get_vocab_size()):
    s=tokenizer.decode([i])
    if any(c in s for c in ["â","Ð","Ñ","¶","±","�"]):
        bad_token_ids.append(i)

bad_token_ids=torch.tensor(bad_token_ids,device=device)

while True:
    x=input("Enter Prompt / Initial tokens: ")
    initial_tokens=tokenizer.encode(x).ids
    initial_tokens=torch.tensor(initial_tokens,dtype=torch.long,device=device).unsqueeze(0)

    generated_tokens=GPT.generate(model,initial_tokens,max_new_tokens=128)[0].tolist()
    generated_text=tokenizer.decode(generated_tokens) 
    x=generated_text.replace("Ġ","").replace("Ċ","\n")
    x=re.sub(
    r'\n',
    lambda _: "" if random.random()<0.8 else "\n",
    x)
    x=re.sub(r'\n\s*\n+','\n',x)
    x="\n".join(line.lstrip() for line in x.splitlines())
    print("Generated Text: ")
    print(x)



