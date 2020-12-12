from collections import Counter


class Corpus:
    def __init__(self, root):
        self.corpus_path = root #输入语料库地址
        self._words = []
        self._puns = set(u"？?!！·【】、；，。、\s+\t+~@#$%^&*()_+{}|:\"<"
                         u"~@#￥%……&*（）——+{}|：“”‘’《》>`\-=\[\]\\\\;',\./■") #标点符号
        self._vocab = set([]) #词典
        self._states = []  #存储训练语料库的BMSE的状态

    # 判断是否存在标点符号
    def is_puns(self, c):
        """
        判断是否符号
        """
        return c in self._puns

    # 生成字典
    def gen_vocabs(self):
        """
        生成词典
        """
        self._vocab = list(set(self._words)) + [u'<UNK>'] #根据语料库生成字典，最后一个为unk
        return self._vocab

    def read_corpus_from_file(self):
        """
        通过方法读取训练语料，将所有的词填充至_words
        """
        with open(self.corpus_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                self._words.extend([word for word in line.strip().split(' ') if word and not self.is_puns(word)])

    def word_to_states(self, word):
        """
        词对应状态转换
        """
        word_len = len(word) #取的是每一个词的长度
        if word_len == 1:
            self._states.append('S')
        else:
            state = ['M'] * word_len #MM, MMM
            state[0] = 'B' #BM , BMM
            state[-1] = 'E' #BE, BME
            self._states.append(''.join(state))

    def cal_init_state(self):
        """
        计算初始概率,改写逻辑变成统计每一句话的第一个字属于BMES总的哪一类
        S: 0.789  B:0.567 M: 0.00000000001 E: 0.0000000000001
        ['BE S S','S S BME'] state[0]
        """
        init_counts = {'S': 0.0, 'B': 0.0,
                       'M': 0.0, 'E': 0.0}
        for state in self._states: #self._states已经被填充完毕了
            init_counts[state[0]] += 1.0  # 每一个词的第一个字(state[0])属于{B,E,M,S}这4种状态的概率
        words_count = len(self._words)  #计算整个语料库的单词数
        init_state = {k: (v + 1) / words_count for k, v in init_counts.items()}
        return init_state

    def cal_trans_state(self):
        """
        计算状态转移概率,从SBME到SBME的转移概率分别为多少
        """
        trans_counts = {'S': {'S': 0.0, 'B': 0.0, 'M': 0.0, 'E': 0.0},
                        'B': {'S': 0.0, 'B': 0.0, 'M': 0.0, 'E': 0.0},
                        'M': {'S': 0.0, 'B': 0.0, 'M': 0.0, 'E': 0.0},
                        'E': {'S': 0.0, 'B': 0.0, 'M': 0.0, 'E': 0.0}}
        states = ''.join(self._states)  # BESSBESS
        counter = Counter(states)  # Counter的输入就是字符串
        for index in range(len(states)):
            if index + 1 == len(states): continue
            #states[0]=B,states[1]=E,第一次执行完B:{S:0,B:0,M：0，}
            #对于'人'来说，trans_counts['B'],对于下一个字'们'，最终是trans['B']['E'] = 1
            trans_counts[states[index]][states[index + 1]] += 1.0  # 计算SBME到SBME的次数
        trans_state = {k: {kk: (vv + 1) / counter[k] for kk, vv in v.items()} for k, v in
                       trans_counts.items()}  # 表示训练集中由 state1 转移到 state2 的次数。SS（SB、SM，SE）的个数/S的总个数
        return trans_state

    def cal_emit_state(self):
        """
        计算观测概率
        """
        self._vocab = self.gen_vocabs()#['手足无措', '八辛庄村']
        #字典中的word_dict={'S': {'周': 0.0007999178750981566, '略': 9.332375209478494e-06}}
        word_dict = {word: 0.0 for word in ''.join(self._vocab)}#{'手': 0.0, '足': 0.0, '无': 0.0, '措': 0.0}
        #假设隐状态是n,词典的长度是m。emit_counts的形状
        #第一个维度称之为行，第二个维度一般称之为列
        #形状：(n,m)
        emit_counts = {'S': dict(word_dict), 'B': dict(word_dict), 'M': dict(word_dict), 'E': dict(word_dict)}
        states = ''.join(self._states)
        counter = Counter(states)
        for index in range(len(self._states)): #self._states = ['BE','S','S','BE',....],表示的是预料库中每个词的状态
            for i in range(len(self._states[index])):# 比如'BE'，依次拿每一个状态
                #表示的是统计语料库中的每个字属于隐状态BMSE的个数
                emit_counts[self._states[index][i]][self._words[index][i]] += 1
        #表示的是每个字属于BMSE的个数 / 总的BMSE
        emit_state = {k: {kk: (vv + 1) / counter[k] for kk, vv in v.items()} for k, v in
                      emit_counts.items()}  # 观测矩阵，emit_mat[state][char]表示训练集中单字char被标注为state的次数
        return emit_state

    def cal_state(self):
        """
        计算3类状态概率
        """
        for word in self._words:
            self.word_to_states(word)#是将训练集中的_words转为隐藏状态（_states）
        init_state = self.cal_init_state() #计算初始state
        trans_state = self.cal_trans_state() #计算转移概率
        emit_state = self.cal_emit_state()
        return init_state, trans_state, emit_state