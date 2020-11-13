'''
Created on 2020年1月13日

@author: Lichengang
'''

from datetime import date

from framework.entity_parser import ParseKnowledgeBase
from framework.parse_rule_loader import build_parser_from_json
import value
import timeexp
import tagger.regex_tagger as tagger

rule_mapping = {"": ["general"], "zh-CN": ["general","chinese"]}
type_path = {"TIME":"time", "VALUE": "value"}
iniializers = {"TIME": timeexp.time_init, "VALUE": value.value_init}
config_iniializers = {"TIME": timeexp.default_time_conf, "VALUE": value.default_value_conf}
mergers = {"TIME": timeexp.time_nerge, "VALUE" : value.value_nerge}

base_path = "res/"

def _init(entity_type="TIME", lang="zh-CN", user_config_dir={}):
    # 加载规则并初始化
    if entity_type not in type_path:
        raise Exception("Invalid entity type :", entity_type)
    if lang not in rule_mapping:
        raise Exception("Invalid language type :", lang)
    config = config_iniializers[entity_type]()
    for config_key in user_config_dir:
        config[config_key] = user_config_dir[config_key]
    file_list = [base_path + type_path[entity_type] + "/" + token + ".json" for token in rule_mapping[lang]]
    parser = ParseKnowledgeBase(entity_type, lang, config)
    iniializers[entity_type](parser)
    parser = build_parser_from_json(parser, file_list)
    return parser

def tag_entity(sentence, lang="zh-CN"):
    '''
    · 标记文本中的实体
    :param  sentence:   文本中的句子
    :return list[dict] 列表，包含实体的开始位置、结束位置，标签信息
    '''
    parsers = []
    for entity_type in type_path.keys():
        parsers.append(_init(entity_type, lang))
    # 正则匹配，保留最长
    matched_parts = tagger.tag_by_parsers(sentence, parsers)
    matched_parts = tagger.select_longest(sentence, matched_parts)
    node_list = [] 
    for matched in matched_parts:
        entity_dict = {"start" : matched.start, "end" : matched.end, "type" : matched.type, 
                "subtype" : matched.rule.type,"mention" : sentence[matched.start:matched.end]} 
        node_list.append(entity_dict)
    # 合并
    for entity_type in mergers:
        node_list = mergers[entity_type](node_list)
    return node_list

def tag_and_parse_entity(sentence, lang="zh-CN"):
    '''
    · 标记文本中的实体
    :param  sentence:   文本中的句子
    :return list[dict] 列表，包含实体的开始位置、结束位置，标签信息
    '''
    parsers = {}
    for entity_type in type_path.keys():
        parsers[entity_type] = _init(entity_type, lang)
    # 正则匹配，保留最长
    matched_parts = tagger.tag_by_parsers(sentence, parsers.values())
    matched_parts = tagger.select_longest(sentence, matched_parts.values())
    if len(matched_parts) == 0:
        return []
    results = []
    for match_info in matched_parts:
        parser =  parsers[match_info.type]
        result_entity = parser.inference(match_info.match, match_info.rule)
        result = {"start" : match_info.start, "end" : match_info.end, "result": result_entity, 
                  "mention" : sentence[match_info.start:match_info.end]}
        results.append(result)
    for entity_type in mergers:
        results = mergers[entity_type](results)
    return results


def parse_single_entity(mention, type="TIME", lang="zh-CN", user_config = {}):
    '''
    · 解析文本中的实体，得到结果对象。
    :param mention:     实体描述
    :param type:        实体类型信息
    :return dict 列表，包含实体的语义信息
    '''  
    parser = _init(type, lang)
    matched_parts = tagger.tag_entity(mention, parser)
    if len(matched_parts) == 0:
        return None
    results = []
    for match_info in matched_parts:
        result_entity = parser.inference(match_info.match, match_info.rule)
        result = {"start" : match_info.start, "end" : match_info.end, "result": result_entity, 
                  "mention" : mention[match_info.start:match_info.end]}
        results.append(result)
    results = mergers[type](results)
    return results[0]


def tag_and_parse_number(sentence, lang="zh-CN"):
    '''
    · 解析文本中的数值对象。
    :param  sentence:   文本
    :param  lang:       语言
    :return list[dict] 列表，包含实体的语义信息
    '''
    parser = _init(entity_type='VALUE')
    matched_parts = tagger.tag_entity(sentence, parser)
    if len(matched_parts) == 0:
        return []
    results = []
    for match_info in matched_parts:
        result_entity = parser.inference(match_info.match, match_info.rule)
        result = {"start" : match_info.start, "end" : match_info.end, "type" : match_info.rule.type, 
                  "result": result_entity, "mention" : sentence[match_info.start:match_info.end]}
        results.append(result)
    value.value_nerge(results)
    return value.parse(results)

def tag_and_parse_time(sentence, lang="zh-CN", document_time=date.today(), prefer_future=True):
    '''
    · 解析文本中的时间对象。
    :param  sentence:   文本
    :param  lang:       语言
    :param  document_time: 文本的时间
    :param  prefer_future: 是否倾向未来时间
    :return dict 包含实体的语义信息
    '''
    custom_config = {}
    custom_config["document_time"] = document_time
    custom_config["prefer_future"] = prefer_future
    parser = _init(user_config_dir=custom_config)
    matched_parts = tagger.tag_entity(sentence, parser)
    if len(matched_parts) == 0:
        return []
    results = []
    for match_info in matched_parts:
        result_entity = parser.inference(match_info.match, match_info.rule)
        result = {"start" : match_info.start, "end" : match_info.end, "type" : match_info.rule.type, 
                  "result": result_entity, "mention" : sentence[match_info.start:match_info.end]}
        results.append(result)
    results = timeexp.time_nerge(results)
    return timeexp.parse(results, document_time)

# if __name__ == '__main__':
#     time_parser = _init("TIME")
#     value_parser = _init("VALUE")
