from torchtext import data

def process_file(f, encoding='utf8'):
    """初始化函数
    Args:
        f[str] : 原始数据文件
        encoding[str]   : 编码(默认utf8)
    """
    data = []
    tag = []
    for line in f:
        x = ''
        y = ''
        nr_flag = False
        nt_flag = False
        nr_word = ''
        nt_word = ''
        #ignore the ID
        for pair in line.strip().split()[1:]:
            word = pair.split('/')[0]
            #split sentence with token '。'
            if word == u'。' and len(x) > 0:
                lines.append(x+'\n')
                labels.append(y.strip()+'\n')
                x = ''
                y = ''
                continue
            #process nt words
            if pair.startswith('['):
                nt_flag = True
                nt_word = word[1:]
                continue
            if nt_flag:
                if not pair.endswith(']nt'):
                    nt_word += word
                    continue
            # process nr tag
        if pair.endswith('nr'):
                nr_word += word
        elif len(nr_word) > 0:
            x += nr_word
            y += ' B-PER'+' I-PER'*(len(nr_word)-1)
            nr_word = ''
        elif pair.endswith('nt'):
            if pair.endswith(']nt'):
                word = nt_word+word
                nt_flag = False
            x += word
            y += ' B-ORG'+' I-ORG'*(len(word)-1)
        # process ns tag
        elif pair.endswith('ns'):
            x += word
            y += ' B-LOC'+' I-LOC'*(len(word)-1)
        else:
            x += word
            y += ' O'*(len(word))
        if len(x) > 0:
            data.append(x+'\n')
            tag.append(y.strip()+'\n')
    return data, tag

class CnNewsDataset(data.Dataset):
    def __init__(self, fname,is_preprocess=True, batch_size=BATCH_SIZE):
        """
        CnNewsDataset
        """
        self.data , self.tag = process_file(fname)
        self.is_preprocess = is_preprocess
        self.batch_size  = batch_size
        
    def build_examples(self):
        fields = [('text', self.text_field), ('label', self.label_field)]
        examples = [data.Example.fromlist([self.data[i], self.tag[i]],
                                    fields) for i in range(len(self.data))]
        super(CnNewsDataset, self).__init__(examples, fields)  
        
    def split_data(self):
        train_data, left_data = self.split(0.3)
        dev_data, test_data = left_data.split(0.3)
        self.text_field.build_vocab(train_data)
        self.label_field.build_vocab(train_data)
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
    
    def get_data_iter(self):
        train_iter, dev_iter, test_iter = data.BucketIterator.splits(
            (self.train_data, self.dev_data, self.test_data),
            batch_sizes=(self.batch_size, 500, 500),
            sort_key=lambda x: len(x.text),
            sort_within_batch=True)
        return train_iter, dev_iter, test_iter









