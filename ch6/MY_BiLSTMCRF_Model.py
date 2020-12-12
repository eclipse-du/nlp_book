import torch
import torch.nn as nn
import torch.nn.functional as F
import nn.utils.rnn.pack_padded_sequence as pack_padded_sequence
import nn.utils.rnn.pad_packed_sequence as pad_packed_sequence
import itertools.zip_longest as zip_longest

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MY_BiLSTMCRF_Model(nn.Module):
    def __init__(self, num_labels, batch_size,
                hidden_dim, vocab_size, embedding_dim, tag2id):
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
        self.transitions = nn.Parameter(
            torch.randn(self.num_labels, self.num_labels))
        self.start_id = tag2id['<sos>']
        self.end_id = tag2id['<eos>']
        self.pad_id = tag2id['<pad>']
        self.tag2id = tag2id


    def cal_logits(self, sentences, mask, lengths):
        embeds = self.word_embeds(sentences)
        lengths_list = lengths.tolist()
        packed = pack_padded_sequence(embeds, lengths_list, batch_first=True)
        packed, _ = self.lstm(packed)
        num_lables = self.num_labels
        sequence_output, _ = pad_packed_sequence(
            packed, batch_first=True)
        # lstm_logits shape:
        # [batch_size, max_len, num_labels, num_labels]
        lstm_logits = self.classifier(sequence_output)
        lstm_logits = lstm_logits.unsqueeze(2)
        lstm_logits = lstm_logits.expand(-1, -1, num_lables, -1)
        # trans_logits shape:
        # [1, num_labels, num_labels]
        trans_logits = self.transitions.unsqueeze(0)
        # logits shape:
        # [batch_size, max_len, num_labels, num_labels]
        logits = lstm_logits + trans_logits
        return logits

    
    def cal_loss(self, logits, targets, mask, lengths):
        batch_size, max_len = targets.shape
        num_labels = self.num_labels
        target_trans = self.indexed(targets)
        # use mask to get real target
        # shape [unmask_length]
        target_trans_real = target_trans.masked_select(mask)
        m_mask = mask.view(batch_size, max_len, 1, 1)
        m_mask = m_mask.expand(batch_size, max_len,
                                num_labels, num_labels)
        logits_real = logits.masked_select(m_mask)
        logits_real = logits_real.view(-1, num_labels*num_labels)
        # get real score for model score index with golden target
        final_scores = logits_real.gather(
            dim=1, index=target_trans_real.unsqueeze(1)).sum()
        all_path_scores = self.get_path_score(logits, lengths)
        loss = (all_path_scores - final_scores) / batch_size
        return loss

    def indexed(self, targets):
        """
        将targets中的数转化为在[T*T]大小序列中的索引
        T是标注的种类
        """
        _, max_len = targets.size()
        for col in range(max_len-1, 0, -1):
            targets[:, col] += (targets[:, col-1] * self.num_labels)
        targets[:, 0] += (self.start_id * self.num_labels)
        return targets

    def get_path_score(self, logits, lengths):
        batch_size, max_len, _, _ = logits.shape
        # scores_t代表batch个字符对应所有标签的分数
        scores_t = torch.zeros(self.batch_size, self.num_labels)
        scores_t = scores_t.to(device)
        for t in range(max_len):
            # 判断batch长度t的样本数量
            batch_size_t = (lengths > t).sum().item()
            if t == 0:
                # 初始化分数由start_id来获取
                scores_t[:batch_size_t] = logits[:batch_size_t, 0, self.start_id, :]
            else:
                # 把原先的前一字符的分数加入当前字符分数列中
                scores_t[:batch_size_t] = torch.logsumexp(
                    logits[:batch_size_t, t, :, :] +
                    scores_t[:batch_size_t].unsqueeze(2),
                    dim=1
                )
        all_path_scores = scores_t[:, self.end_id].sum()
        return all_path_scores

    
    def forward(self, sentences, mask, targets):
        lengths = torch.sum(mask, dim=1)
        logits = self.cal_logits(sentences, mask, lengths)
        return self.cal_loss(logits, targets, mask, lengths)

    def predict(self, sentences, mask):
        # calculate crf logits score
        lengths = torch.sum(mask, dim=1)
        logits = self.cal_logits(sentences, mask, lengths)
        result = self.viterbi(logits, lengths)
        return result
    
    def viterbi(self, logits, lengths):
        batch_size, max_len, num_labels, _ = logits.shape 
        viterbi = torch.zeros(batch_size, max_len, num_labels)
        viterbi = viterbi.to(device)
        bp = torch.zeros(batch_size, max_len, num_labels).long()
        backpointer = (bp * self.end_id)
        backpointer = backpointer.to(device)
        # convert length into long tensor
        lengths = lengths.long().to(device)
        # 向前递推
        for step in range(max_len):
            batch_size_t = (lengths > step).sum().item()
            if step == 0:
                viterbi[:batch_size_t, step,:] = logits[: batch_size_t, step, self.start_id, :]
                backpointer[: batch_size_t, step, :] = self.start_id
            else:
                max_scores, prev_tags = torch.max(
                    viterbi[:batch_size_t, step-1, :].unsqueeze(2) +
                    logits[:batch_size_t, step, :, :],dim=1
                )
                viterbi[:batch_size_t, step, :] = max_scores
                backpointer[:batch_size_t, step, :] = prev_tags 
        backpointer = backpointer.view(batch_size, -1)  # [B, L * T]
        tagids = []  # 存放结果
        tags_t = None
        for step in range(max_len-1, 0, -1):
            batch_size_t = (lengths > step).sum().item()
            if step == max_len-1:
                index = torch.ones(batch_size_t).long() * (step * num_labels)
                index = index.to(device)
                index += self.end_id
            else:
                prev_batch_size_t = len(tags_t)
                before_batch_value = (batch_size_t - prev_batch_size_t)
                new_in_batch = [self.end_id] * before_batch_value
                new_in_batch = torch.LongTensor(new_in_batch).to(device)
                offset = torch.cat([tags_t, new_in_batch], dim=0)
                tmp = torch.ones(batch_size_t).long()
                index = tmp * (step * self.num_labels)
                index = index.to(device)
                index += offset.long()
            tags_t = backpointer[:batch_size_t].gather(
                dim=1, index=index.unsqueeze(1).long())
            tags_t = tags_t.squeeze(1)
            tagids.append(tags_t.tolist())
        tagids = list(zip_longest(*reversed(tagids), fillvalue=self.pad_id))
        tagids = torch.Tensor(tagids).long()
        return tagids






