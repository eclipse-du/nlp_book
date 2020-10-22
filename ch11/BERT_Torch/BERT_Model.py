"""
@author: liucong
@contact: logcongcong@gmail.com
@time: 2019/11/10 10:14
"""
from BERT.transformers.transformers.modeling_bert import BertPreTrainedModel, BertModel, BertOnlyNSPHead


class BERT(BertPreTrainedModel):
    def __init__(self, bert_config):
        """
        BERT模型初始化
        :param bert_config: bert原始参数
        """
        super(BERT, self).__init__(bert_config)
        self.bert = BertModel(bert_config)
        self.cls = BertOnlyNSPHead(bert_config)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        seq_relationship_score = self.cls(pooled_output)
        outputs = seq_relationship_score
        return outputs