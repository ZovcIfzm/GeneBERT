from transformers import BertTokenizer, BertModel
import torch

class Net(torch.nn.Module): 
    def init(self): 
        super(Net, self).init() 
        self.bert = BertModel.from_pretrained('./text')
        # Uncomment below if you wanna freeze bert
        #for p in self.bert.parameters():
        #    p.requires_grad = False
        self.fc0 = torch.nn.Linear(768, 1)
    def forward(self,x,att,pos): # attention mask, position of [mask] token
        hidden = self.bert(x,attention_mask=att)[0]
        net = hidden[torch.arange(hidden.shape[0]),pos,:]
        net = self.fc0(net)
        return net