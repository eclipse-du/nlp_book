#! /usr/bin/env python
# coding:utf-8
'''
Created on 2019年11月29日
@copyright: code of chapter 12
@author: Lichengang
'''

from framework.entity_parser import ParseKnowledgeBase, FunctionOperator
from value import number_parser

def create_config():
    return {}

def operate_norm_number(operand : list, config):
    ''' 数值操作符
    '''
    assert len(operand) >= 1
    text = operand[0]
    value = number_parser.norm_arabic_num(text)
    if not value:
        value = number_parser.norm_chinese_num(text)
    return str(value) if value else ""

def operate_chinese_num(operand : list, config):
    ''' 中文数值操作符
    '''
    return number_parser.norm_chinese_num(operand[0])

def operate_add(operand : list, config):
    ''' 加法操作符。用于带分数
    '''
    assert len(operand) >= 2
    return str(float(operand[0]) + float(operand[1]))

def operate_div(operand : list, config):
    ''' 除法操作符。用于分数
    '''
    assert len(operand) >= 2
    return str(float(operand[0]) / float(operand[1]))

def operate_dot(operand : list, config):
    ''' 小数点操作符
    '''
    assert len(operand) >= 2
    return str(int(operand[0]) + int(operand[1]) / pow(10, len(operand[1])))

def operate_max(operand : list, config):
    ''' 最大值操作符
    '''
    assert len(operand) >= 1
    copy = list(operand[0])
    end_zero = True
    for i in range(len(operand[0]), 0, -1):
        if end_zero:
            if operand[0][i-1] != '0':
                end_zero = False
            else:
                copy[i-1] = '9'
    return "".join(copy)

def operate_level(operand : list, config):
    ''' 数量级操作符
    '''
    assert len(operand) >= 2
    value = operate_norm_number(operand[0:1], config)
    if value == "":
        return 0
    return str(number_parser.norm_level(int(value), operand[1][0]))

def init(parser : ParseKnowledgeBase):
    parser.add_operator(FunctionOperator("Add", operate_add))
    parser.add_operator(FunctionOperator("Div", operate_div))
    parser.add_operator(FunctionOperator("Dot", operate_dot))
    parser.add_operator(FunctionOperator("Level", operate_level))
    parser.add_operator(FunctionOperator("Max", operate_max))
    parser.add_operator(FunctionOperator("NormNumber", operate_norm_number))

if __name__ == '__main__':
    print(operate_add(["5","0.5"], create_config() ))
    print(operate_div(["15","100"], create_config() ))
    print(operate_dot(["5","18"], create_config() ))
    print(operate_max(["15000"], create_config() ))
    print(operate_level(["5", "亿"], create_config() ))