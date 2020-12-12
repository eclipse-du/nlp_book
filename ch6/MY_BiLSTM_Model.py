import torch
import torch.nn as nn
import torch.nn.functional as F
import nn.utils.rnn.pack_padded_sequence as pack_padded_sequence
import nn.utils.rnn.pad_packed_sequence as pad_packed_sequence

class MY_BiLSTM_Model(nn.Module):
    def __init__(self, num_labels, batch_size,
                hidden_dim, vocab_size, embedding_dim):
        super().__init__()
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_labels = num_labels
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,
                            bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_labels)

    def cal_logits(self, sentences, mask):
        embeds = self.word_embeds(sentences)
        lengths = torch.sum(mask, dim=1).tolist()
        packed = pack_padded_sequence(embeds, lengths, batch_first=True)
        packed, _ = self.lstm(packed)
        sequence_output, _ = pad_packed_sequence(
            packed, batch_first=True)
        return self.classifier(sequence_output)
    
    def cal_loss(self, logits, targets, mask):
        targets = targets[mask]
        m = mask.unsqueeze(2).expand(-1, -1, self.num_labels)
        logits = logits.masked_select(m)
        logits = logits.view(-1, self.num_labels)
        loss = F.cross_entropy(logits, targets)
        return loss

    def forward(self, sentences, mask, targets):
        logits = self.cal_logits(sentences, mask)
        return self.cal_loss(logits, targets, mask)

    def predict(self, sentences, mask):
        logits = self.cal_logits(sentences, mask)
        _, result = torch.max(logits, dim=2)
        return result




