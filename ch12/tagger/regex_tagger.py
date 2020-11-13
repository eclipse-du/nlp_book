#! /usr/bin/env python
# coding:utf-8
'''
Created on 2019年11月13日
@copyright: code of chapter 12
@author: Lichengang
'''

from framework.entity_parser import ParseKnowledgeBase

class MatchResult(object):
    '''
    · 文本匹配的结果
    '''
    def __init__(self, match, rule, start=-1, end=-1):
        # 正则的匹厄结果
        self.match = match
        self.rule = rule
        # 开始位置
        self.start = start
        if self.start == -1:
            start = match.start()
        # 结束位置
        self.end = end
        if self.end == -1:
            end = match.end()
        self.type = ""    
            
    def __len__(self):
        return self.end - self.start

def select_longest(text : str, result_list : list):
    '''
    · 选择最长的实体，作为结果
    '''
    # MatchResult 构造的 DAG，下标为开始位置，值为长度
    result_list.sort(key = lambda x : len(x), reverse = True)
    graph = [None] * len(text)
    selected_list = []
    for item in result_list:
        for i in range(item.start, item.end):
            if graph[i] is not None:
                break
        else:
            graph[item.start:item.end] = [item]*len(item)
            selected_list.append(item)
    return selected_list

def tag_entity(text : str, entity_knowledge_base : ParseKnowledgeBase):
    '''
    · 在文本中标注片段
    '''
    result_list = []
    for rule in entity_knowledge_base.rules.values():
        for match in rule.regex.finditer(text):
            result_list.append(MatchResult(match, rule, match.start(), match.end()))
    return select_longest(text, result_list)

def tag_by_parsers(text : str, parser_list : list):
    '''
    · 在文本中用规则标注
    '''
    result_list = []
    for parser in parser_list:
        for rule in parser.rules.values():
            for match in rule.regex.finditer(text):
                if match.end() > match.start():
                    matched = MatchResult(match, rule, match.start(), match.end())
                    matched.type = parser.type
                    result_list.append(matched)
    return result_list

from framework.parse_rule_loader import build_parser_from_json

if __name__ == '__main__':
    # 测试解析数据集
    parser = build_parser_from_json(ParseKnowledgeBase("timeexp"), 
                                    ["../../res/time/general.json"])
    selected_list = tag_entity("2019-06-30", parser)
    for part in selected_list:
        if part is not None:
            print("%s %s %s %s" % (part.start, part.end, part.rule, part.match))

    