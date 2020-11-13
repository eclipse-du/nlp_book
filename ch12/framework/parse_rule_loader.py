#! /usr/bin/env python
# coding:utf-8
'''
Created on 2019年10月27日
@copyright: code of chapter 12
@author: Lichengang
'''

import codecs
import os
import re
import json
import logging
from framework.entity_parser import ParseKnowledgeBase, ParserRule, WordGroup, Operation, ReplaceOperator

logger = logging.getLogger(__name__)

def count_parentheses(pattern_string):
    ''' 统计括号对的数量
    :param pattern_string:  字符串
    :return: int 括号对不匹配返回 -1，否则返回括号对数
    '''
    layer = 0
    group_count = 0
    last_ch = ' '
    for ch in pattern_string:
        if last_ch == '\\':
            last_ch = ch
            continue
        if ch == '(':
            layer += 1
        elif ch == ')':
            layer -= 1
            group_count += 1
        last_ch = ch
    if layer != 0:
        return -1
    return group_count

def count_left_parenthes(pattern_string):
    ''' 统计左括号的数量
    :param pattern_string:  字符串
    :return: int 左括号数量
    '''
    group_count = 0
    if len(pattern_string) == 0:
        return group_count
    last_ch = ' '
    for ch in pattern_string:
        if last_ch == '\\':
            pass
        elif ch == '(':
            group_count += 1
    return group_count

def is_valid_re(pattern):
    try:
        regex = re.compile(pattern)
        return regex
    except re.error:
        return None

def build_parser_from_json(parser : ParseKnowledgeBase, paths : list):
    ''' 加载JSON格式的文件，返回结果
    :param parser:  ParseKnowledgeBase 类对象
    :param paths:   单个或多个文件
    :return:        parser
    '''
    for path in paths:
        path = os.path.abspath(path)
        with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            if content.startswith(u'\ufeff'):
                content = content[1:]
            json_object = json.loads(content)
            # 1.加载词库
            vocabulary = {}
            word_group_list = json_object["vocabulary"]
            for word_group_dict in word_group_list:
                word_group = WordGroup(word_group_dict["name"], word_group_dict["value"]);
                add_word(word_group, vocabulary)
            # 2.加载替换式算子
            operator_list = json_object["normalizers"]
            for operator_dict in operator_list:
                new_operator = ReplaceOperator(operator_dict['name'], operator_dict['value'])
                parser.add_operator(new_operator)
            # 3.加载匹配规则
            rule_list = json_object["rules"]
            for rule_dict in rule_list:
                type = rule_dict["type"] if "type" in rule_dict else ""
                rule = ParserRule(rule_dict["ruleName"], type, rule_dict["extraction"])
                build_regex_pattern(rule, vocabulary)
                normalization_rules = rule_dict['normalization']
                if count_parentheses(normalization_rules) < 0:
                    logger.error("rule %s has an invalid normalization pattern." % rule.name)
                else:
                    rule.operations = parse_operation(normalization_rules)
                    if not parser.verify_operation(rule.operations):
                        logger.error("normalization rule %s has unknown operators" % normalization_rules)
                    parser.add_rule(rule)
    return parser

def build_regex_pattern(rule : ParserRule, vocabulary):
    ''' 检查匹配规则，并从规则生成正则表达式 Pattern
    :param rule:        规则, ParserRule对象
    :param vocabulary:  加载好的词库，dict形式
    '''
    gruop_mapping = [0] # 分组偏移表
    delta = 0 # 分组偏移数量
    last_end = 0 # 当前结尾
    res_pattern = re.compile(r'\$<([A-Za-z]\w+)>')
    final_pattern = rule.pattern_string
    for m in res_pattern.finditer(rule.pattern_string):
        manual_parenthes_count = count_left_parenthes(rule.pattern_string[last_end:m.start()])
        curgroup = len(gruop_mapping)
        for i in range(manual_parenthes_count):
            gruop_mapping.append(curgroup+i+delta)
        vocab_name = m.group(1)
        if vocab_name in vocabulary:
            delta += count_parentheses(vocabulary[vocab_name].pattern)
            final_pattern = final_pattern.replace(m.group(), vocabulary[vocab_name].pattern, 1)
        else:
            logger.warning("cannot find %s in vocabulary!" % vocab_name)
        gruop_mapping.append(len(gruop_mapping)+delta)
        last_end = m.end()
    # 测试正则
    parenthes_count = count_parentheses(final_pattern)
    if (parenthes_count == -1):
        logger.error("cannot build %s， invalid parentheses align。" % rule.name)
        return None
    regex = is_valid_re(final_pattern)
    if regex is None:
        logger.error("cannot build %s， invalid final regular expression。" % rule.name)
        return None
    rule.regex = regex
    rule.parenthes_count = parenthes_count
    rule.gruop_mapping = gruop_mapping
    return rule

def validate_word(word):
    invalid = is_valid_re(word) is None
    if invalid:
        logger.warning("word %s is illegal, will skip" % word)
    return not invalid

def add_word(word_group, vocabulary):
    ''' 检查词库， 并且将词库构建为一个局部Pattern
    :param word_group:  词语列表
    :param vocabulary:  词库
    :return: int 括号对不匹配返回 -1，否则返回括号对数
    '''
    valid_list = [word for word in word_group.word_list if validate_word(word)]
    vocab_pattern = "(" + '|'.join(valid_list) + ")"
    word_group.pattern = vocab_pattern
    vocabulary[word_group.name] = word_group

def parse_operation(operation_text):
    '''
    · 解析操作符
    :param operation_text:
    :return list 解析结果，列表形式
    '''
    operator_pattern = r"%([0-9A-Za-z]+)\s*\(\s*"
    # 无名称变量下标
    var_idx = 0
    
    # 最终结果
    operations = []
    # 0-字符串状态 1-算子/操作数token 2-token结束的空格 3-操作数分隔符 
    is_in_operation = False
    # 算子堆栈和状态堆战，分别保存了当前part的内容
    cur_operation = None
    operation_stack = []
    # 嵌套算子展开后的列表
    operation_list = []
    idx = 0
    token = ""
    while idx < len(operation_text):
        if is_in_operation:
            if ' ' == operation_text[idx] or '\t' == operation_text[idx]:
                pass
            elif operation_text[idx] == '(':
                # 前进一层
                cur_operation = Operation(token)
                operation_stack.append(cur_operation)
                token = ''
            elif operation_text[idx] == ',':
                if not token == '':
                    cur_operation.operand.append(token)
                token = ''
            elif operation_text[idx] == ')':
                # 弹出一层
                cur_operation = operation_stack.pop()
                if not token == '':
                    cur_operation.operand.append(token)
                token = ''
                return_variable = "__var_%d" % var_idx
                var_idx += 1
                cur_operation.result = return_variable
                operation_list.append(cur_operation)
                if (len(operation_stack) == 0):
                    operations.append(operation_list)
                    operation_list = []
                    is_in_operation = False
                else:
                    cur_operation = operation_stack[-1]
                    cur_operation.operand.append(return_variable)
            else:
                token += operation_text[idx]
            idx += 1
        else:
            # 识别算子
            match = re.search(operator_pattern, operation_text[idx:], re.ASCII)
            if match:
                operator_name = match.group(1)
                # 前导字符串
                token = ''
                plain_text = operation_text[idx:idx+match.start(0)]
                if plain_text != "":
                    operations.append(plain_text)
                # 创建操作符
                cur_operation = Operation(operator_name)
                operation_stack.append(cur_operation)
                is_in_operation = True
                idx += match.end(0)
            else:
                operations.append(operation_text[idx:])
                break
    if len(operation_stack) > 0:
        logger.error('\"%s\" is not a valid expression!' % operation_text)
    return operations
    
if __name__ == '__main__':
    # 测试解析单条实体解析计算规则
    print(parse_operation("P%NormNumber(group(1))%Unit4Duration(group(2))x"))
    print(parse_operation("%Add( group(1), Div(NormNumber(group(3)), NormNumber(group(2))) )"))
    # 测试解析数据集
    parser = build_parser_from_json(ParseKnowledgeBase("timeexp"), ["res/value/chinese.json"])


