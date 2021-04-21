

class BertForSequenceClassification(nn.Module):
  
    def __init__(self, num_labels=2):
        super(BertForSequenceClassification, self).__init__()        
        self.num_labels = num_labels        
        self.bert = BertModel.from_pretrained('bert-base-uncased')        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)        
        self.classifier = nn.Linear(config.hidden_size, num_labels)        
        nn.init.xavier_normal_(self.classifier.weight)    

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):        
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)        
        pooled_output = self.dropout(pooled_output)        
        logits = self.classifier(pooled_output)

        return logits

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokenized_text = tokenizer.tokenize(some_text)
tokenizer.convert_tokens_to_ids(tokenized_text)

max_seq_length = 256
class text_dataset(Dataset):
    def __init__(self,x_y_list):self.x_y_list = x_y_list
        
    def __getitem__(self,index):
        
        tokenized_review = tokenizer.tokenize(self.x_y_list[0][index])
        
        if len(tokenized_review) > max_seq_length:
            tokenized_review = tokenized_review[:max_seq_length]
            
        ids_review  = tokenizer.convert_tokens_to_ids(tokenized_review)

        padding = [0] * (max_seq_length - len(ids_review))
        
        ids_review += padding
        
        assert len(ids_review) == max_seq_length
        
        #print(ids_review)
        ids_review = torch.tensor(ids_review)
        
        sentiment = self.x_y_list[1][index] # color        
        list_of_labels = [torch.from_numpy(np.array(sentiment))]
        
        
        return ids_review, list_of_labels[0]
    
    def __len__(self):
        return len(self.x_y_list[0])