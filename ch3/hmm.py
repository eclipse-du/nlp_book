import numpy as np
from hmmlearn.hmm import MultinomialHMM


class Segment:

    def __init__(self,util):
        self.util = util
        self.states, self.init_p = self.get_init_state()
        self.trans_p = self.get_trans_state()
        self.vocabs, self.emit_p = self.get_emit_state()
        self.model = self.get_model()

    def get_init_state(self):
        """
        获取初始概率，转为hmm模型接受数据形式
        """
        states = ['S', 'B', 'M', 'E']
        init_state = self.util.read_state_from_file('./init_stats.txt')
        init_p = np.array([init_state[s] for s in states])
        return states, init_p

    def get_trans_state(self):
        """
        获取转移概率，转为hmm模型接受数据形式
        """
        trans_state = self.util.read_state_from_file('./trans_stats.txt')
        trans_p = np.array([[trans_state[s][ss] for ss in self.states] for s in self.states])
        return trans_p

    def get_emit_state(self):
        """
        获取发射概率，转为hmm模型接受数据形式
        """
        emit_state = self.util.read_state_from_file('./emit_stats.txt')
        vocabs = []
        for s in self.states:
            vocabs.extend([k for k, v in emit_state[s].items()])
        vocabs = list(set(vocabs))
        emit_p = np.array([[emit_state[s][w] for w in vocabs] for s in self.states])
        return vocabs, emit_p

    def get_model(self):
        """
        初始化hmm模型
        """
        model = MultinomialHMM(n_components=len(self.states))
        model.startprob_ = self.init_p
        model.transmat_ = self.trans_p
        model.emissionprob_ = self.emit_p
        return model

    def pre_process(self, word):
        """
        未知字处理
        """
        if word in self.vocabs:
            return self.vocabs.index(word)
        else:
            return len(self.vocabs)-1

    def cut(self, sentence):
        """
        分词
        """
        seen_n = np.array([[self.pre_process(w) for w in sentence]]).T
        log_p, b = self.model.decode(seen_n, algorithm='viterbi')
        states = map(lambda x: self.states[x], b)
        cut_sentence = ''
        index =0
        for s in states:
            if s in ('S', 'E'):
                cut_sentence += sentence[index]+' '
            else:
                cut_sentence += sentence[index]
            index += 1
        return cut_sentence