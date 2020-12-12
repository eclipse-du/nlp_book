# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from cut import Corpus
from hmm import Segment
from util import Utils

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    '''
    
    s = 'sssmmmee'
    another_s = ['sir','ms','sir']
    from collections import Counter
    counter = Counter(s)
    another_counter = Counter(another_s)
    print(counter)
    print(another_counter)
    '''
    
    corpus = Corpus('./msr_training.utf8')
    corpus.read_corpus_from_file()
    init_state, trans_state, emit_state = corpus.cal_state()
    #print(emit_state)
    #util = Utils()
    #util.save_state_to_file('./init_stats.txt', init_state)
    #util.save_state_to_file('./trans_stats.txt', trans_state)
    #util.save_state_to_file('./emit_stats.txt', emit_state)
    #seg = Segment(util)
    #print(seg.cut('中国的人工智能发展进入高潮阶段'))
