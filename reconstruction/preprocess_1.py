#public packages
import json
import copy
import pdb
import os
from typing import List, Tuple, Dict, Set
from copy import deepcopy
import torch
# import joblib

#user source code
from utils.parameter_setting import args,PAD_token,SOS_token,EOS_token,UNK_token,GATES,REVERSE_GATES,USE_CUDA
from utils.Vocab import Vocabs
# from utils.dataset import *
def get_slots_information(ontology:dict)->Tuple[Tuple[str],Set[str]]:
    #description:获取所有领域和领域-槽
    ALL_SLOTS = tuple(k.replace(" ","").lower() for k in ontology.keys())
    ALL_DOMAINS = set([e.split('-')[0] for e in ALL_SLOTS])
    return ALL_SLOTS,ALL_DOMAINS

def fix_label_value_error(labels,ALL_SLOTS):
    #处理掉一些标签值上的错误
    #label:[{slots:[[name,value]],act:'inform'},...]
    label_dict = {l["slots"][0][0]:l["slots"][0][1] for l in labels}
    GENERAL_TYPO = {# type
        "guesthouse":"guest house", "guesthouses":"guest house", "guest":"guest house", "mutiple sports":"multiple sports", 
        "sports":"multiple sports", "mutliple sports":"multiple sports","swimmingpool":"swimming pool", "concerthall":"concert hall", 
        "concert":"concert hall", "pool":"swimming pool", "night club":"nightclub", "mus":"museum", "ol":"architecture", 
        "colleges":"college", "coll":"college", "architectural":"architecture", "musuem":"museum", "churches":"church",
        # area
        "center":"centre", "center of town":"centre", "near city center":"centre", "in the north":"north", "cen":"centre", "east side":"east", 
        "east area":"east", "west part of town":"west", "ce":"centre",  "town center":"centre", "centre of cambridge":"centre", 
        "city center":"centre", "the south":"south", "scentre":"centre", "town centre":"centre", "in town":"centre", "north part of town":"north", 
        "centre of town":"Acentre", "cb30aq": "none",
        # price
        "mode":"moderate", "moderate -ly": "moderate", "mo":"moderate", 
        # day
        "next friday":"friday", "monda": "monday", 
        # parking
        "free parking":"free",
        # internet
        "free internet":"yes",
        # star
        "4 star":"4", "4 stars":"4", "0 star rarting":"none",
        # others 
        "y":"yes", "any":"dontcare", "n":"no", "does not care":"dontcare", "not men":"none", "not":"none", "not mentioned":"none",
        '':"none", "not mendtioned":"none", "3 .":"3", "does not":"no", "fun":"none", "art":"none",  
    }
    for slot in ALL_SLOTS:
        if slot not in label_dict.keys():
            continue
        # 槽值简写恢复
        label_dict[slot] = GENERAL_TYPO.get(label_dict[slot],label_dict[slot])
        # 槽值匹配不上，槽值根本不在ontology范围内，或者语义上说不通
        if  slot == "hotel-type" and label_dict[slot] in ["nigh", "moderate -ly priced", "bed and breakfast", "centre", "venetian", "intern", "a cheap -er hotel"] or \
            slot == "hotel-internet" and label_dict[slot] == "4" or \
            slot == "hotel-pricerange" and label_dict[slot] == "2" or \
            slot == "attraction-type" and label_dict[slot] in ["gastropub", "la raza", "galleria", "gallery", "science", "m"] or \
            "area" in slot and label_dict[slot] in ["moderate"] or \
            "day" in slot and label_dict[slot] == "t":
            label_dict[slot] = "none"
        elif slot == "hotel-type" and label_dict[slot] in ["hotel with free parking and free wifi", "4", "3 star hotel"]:
            label_dict[slot] = "hotel"
        elif slot == "hotel-star" and label_dict[slot] == "3 star hotel":
            label_dict[slot] = "3"
        elif "area" in slot:
            if label_dict[slot] == "no": label_dict[slot] = "north"
            elif label_dict[slot] == "we": label_dict[slot] = "west"
            elif label_dict[slot] == "cent": label_dict[slot] = "centre"
        elif "day" in slot:
            if label_dict[slot] == "we": label_dict[slot] = "wednesday"
            elif label_dict[slot] == "no": label_dict[slot] = "none"
        elif "price" in slot and label_dict[slot] == "ch":
            label_dict[slot] = "cheap"
        elif "internet" in slot and label_dict[slot] == "free":
            label_dict[slot] = "yes"
        # some out-of-define classification slot values
        if  slot == "restaurant-area" and label_dict[slot] in ["stansted airport", "cambridge", "silver street"] or \
            slot == "attraction-area" and label_dict[slot] in ["norwich", "ely", "museum", "same area as hotel"]:
            label_dict[slot] = "none"
        if slot == 'hotel-name' and label_dict[slot] in ["no", "yes"]:
            label_dict[slot] = "none"
        if slot == 'restaurant-name' and label_dict[slot] in ["no", "yes"]:
            label_dict[slot] = "none"
    return label_dict

def fix_domainSlot_name(turn_belief_dict, ALL_SLOTS):
    #将标签中的domain-slot中包含的空格去掉
    out = {}
    for k in turn_belief_dict.keys():
        new_k = k.replace(" ", "")
        if new_k not in ALL_SLOTS: pdb.set_trace()
        out[new_k] = turn_belief_dict[k]
    return out

def rule_out_multiVal(turn_belief_dict,multival_counter):
    #将多值转化为单值
    has_multival = False
    for k,v in turn_belief_dict.items():
        if '|' in v:
            turn_belief_dict[k] =  v.split('|')[0] #' ; '.join(values)
            has_multival = True
    if has_multival: multival_counter += 1 
    return turn_belief_dict, multival_counter

def get_normalized_value_len(ordered_domainSlots, turn_belief_dict, slot_gating):
    #标签值长度向量化:
    normalized_val_len = [0] * len(ordered_domainSlots)
    if slot_gating:
        gates = [GATES['none']] * len(ordered_domainSlots)
    else:
        gates = None
    for k,v in turn_belief_dict.items():
        index = ordered_domainSlots.index(k)
        lenval = len(v.split())
        if not slot_gating or (slot_gating and v not in ['dontcare', 'none']):
            normalized_val_len[index] = lenval
        if slot_gating:
            if v not in ['dontcare', 'none']:
                gates[index] = GATES['gen']
            else:
                gates[index] = GATES[v]
    return normalized_val_len, gates

def get_nonatrg_normalized_y(ordered_domainSlots, normalized_val_len, turn_belief_dict):
    #返回不定长的序列
    normalized_domains_seq = []
    normalized_slots_seq = []
    normalized_domainslot_index_seq = []
    normalized_val_seq = []
    for idx, lenval in enumerate(normalized_val_len):
        if lenval == 0: continue 
        domain = ordered_domainSlots[idx].split('-')[0]
        slot = ordered_domainSlots[idx].split('-')[1]
        val = turn_belief_dict[ordered_domainSlots[idx]].split()
        assert(len(val) == lenval)
        for i in range(lenval):
            normalized_domains_seq.append(domain+"_DOMAIN")
            normalized_slots_seq.append(slot+"_SLOT")
            normalized_val_seq.append(val[i])
            normalized_domainslot_index_seq.append(idx)
    return normalized_domains_seq, normalized_slots_seq, normalized_val_seq, normalized_domainslot_index_seq

def get_atrg_normalizde_y(ordered_domainSlots, normalized_val_len, turn_belief_dict):
    vals = []
    indices = []
    for idx, lenval in enumerate(normalized_val_len):
        if lenval == 0: continue 
        val = turn_belief_dict[ordered_domainSlots[idx]].split()
        assert(len(val) == lenval)
        vals.append(val)
        indices.append(idx)
    return vals, indices

def extract_dialogues(data, ALL_SLOTS, global_vocab, mem_vocab, is_train,args):
    processed_data=[]#存储预处理的结果
    ordered_domainSlots=sorted(ALL_SLOTS)
    ordered_domains, ordered_slots = [], []
    for i in ordered_domainSlots:
        tmp=i.split('-')
        ordered_domains.append(tmp[0]+'_DOMAIN')
        ordered_slots.append(tmp[1]+'_SLOT')
    #记录domain-slot可取值的最长长度与值 tuple(len(value.split(),value)
    max_len_domainSlotValue={ds:(1,'none') for ds in ordered_domainSlots}
    #记录一个槽可取值的最长长度, 也即max(max_len_domainSlotValue.items(),key=lambda x:x[1][0])[1][0]
    max_len_a_domainSlotValue=1
    domain_counter={}#统计每个领域在语料中有多少次对话
    #统计槽值可取多值的对话状态数(一轮次对话有一个状态，一次对话有n轮次)
    multival_counter=0#领域槽含有多个可选值对话状态数,类似于一个领域槽的槽值含有别名
    #遍历所有对话
    for dial in data:
        #领域对话数个更新
        for domain in dial["domains"]:
            domain_counter[domain] = domain_counter.get(domain,0)+1
        dialogue_history=''#对话历史
        delex_dialogue_history=''#将领域槽值替换成领域槽标记的对话历史
        #遍历一次对话的所有轮次
        for tid, turn in enumerate(dial['dialogue']):
            #**********对话状态标签***********
            #处理掉一般性的标签错误，返回槽值对词典slot:value
            turn_belief_dict = fix_label_value_error(turn["belief_state"], ALL_SLOTS)
            #去除domain_slot中含有的空格
            turn_belief_dict = fix_domainSlot_name(turn_belief_dict, ALL_SLOTS)
            #处理标签中可取的多个值的情况, 只留下一个
            turn_belief_dict, multival_counter=rule_out_multiVal(turn_belief_dict,multival_counter)
            #去除None值的槽值对
            turn_belief_dict = {k:v for k,v in turn_belief_dict.items() if v!='none'}
            turn_belief_list = [str(k)+'-'+str(v) for k, v in turn_belief_dict.items()]
            #标签值长度
            normalized_val_len, gates = get_normalized_value_len(ordered_domainSlots, turn_belief_dict, args['slot_gating'])#获取对应槽值长度
            # 生成的领域槽和值序列 length of value * domains- nonautoregressive model
            domains_seq, slots_seq, y_val_seq, domainslots2_index_seq = \
                get_nonatrg_normalized_y(ordered_domainSlots, normalized_val_len, turn_belief_dict)
            domainslots_index_auto=None
            y_val_auto = None
            if args['auto_regressive']:
                #slot_values,domain_slot_index
                y_val_auto, domainslots_index_auto =\
                    get_atrg_normalizde_y(ordered_domainSlots, normalized_val_len, turn_belief_dict)

            #**********标签相关统计值计算********
            #更新槽值最大长度
            for k,v in turn_belief_dict.items():
                tmp=len(v.strip().split())
                if  tmp> max_len_domainSlotValue[k][0]:
                    max_len_domainSlotValue[k] = (tmp, v)
            #**********对话内容数据***********
            user_sent = ' SOS ' + turn["transcript"] + ' EOS '
            dlx_user_sent = ' SOS ' + turn["delex_transcript"] + ' EOS '
            if tid==0:#第一轮次系统没有话语发出
                sys_sent = ''
                dlx_sys_sent = ''
            else:
                sys_sent = ' SOS '  + turn["system_transcript"] + ' EOS '
                dlx_sys_sent = 'SOS ' + turn["delex_system_transcript"] + ' EOS '
            turn_uttr = sys_sent + user_sent 
            dialogue_history += sys_sent +user_sent
            if args['delex_his']:
                in_delex_dialogue_history= delex_dialogue_history+dlx_sys_sent+user_sent
            delex_dialogue_history += dlx_sys_sent+dlx_user_sent
            #**********词典更新**********
            if is_train==True or args['pointer_decoder']:
                #扩充词典条件:训练集或者使用了指针解码器
                global_vocab.index_words(user_sent,'utter')#用户话语
                global_vocab.index_words(sys_sent,'utter')
            if  args['delex_his'] and (is_train or args['pointer_decoder']):
                global_vocab.index_words(dlx_user_sent, 'utter')
                global_vocab.index_words(dlx_sys_sent, 'utter')
            if is_train:
                mem_vocab.index_words(turn_belief_dict, 'belief')
            #**********所有数据聚集**********
            a_turn={#一个轮次对话
                "ID": dial["dialogue_idx"], 
                "turn_id": tid, 
                "dialog_history": dialogue_history.strip(), 
                "delex_dialog_history": in_delex_dialogue_history.strip(),
                "turn_uttr":turn_uttr.strip(), 
                "ordered_domainSlots": ordered_domainSlots,
                'ordered_domains': ordered_domains,
                'ordered_slots': ordered_slots,
                "turn_belief_dict": turn_belief_dict, 
                "turn_belief_list":turn_belief_list,
                'normalized_val_len': normalized_val_len,
                'gates': gates, 
                'y_val_seq': y_val_seq,
                'domains_seq': domains_seq,
                'slots_seq': slots_seq,
                'domainslots_index_auto': domainslots_index_auto, 
                'y_val_auto': y_val_auto
            }
            processed_data.append(a_turn)
    max_len_a_domainSlotValue=max(max_len_domainSlotValue.items(),\
        key=lambda x:x[1][0])[1][0]
    print('the number of turns:',len(processed_data))
    print('the number of dialogue grouped by domain:', domain_counter)
    print('the number of turns belief state containing multi-value: ', multival_counter)
    return processed_data, max_len_a_domainSlotValue, max_len_domainSlotValue

def get_model_input(data, global_vocab, mem_vocab, domain_vocab, slot_vocab, batch_size, shuffle, args, split, ALL_SLOTS):  
    #pair: 对话的数据，包含字典的列表
    #lang: 全局字典 mem_lang：对话状态的字典; domain_lang:领域词典 slot_lang：槽词典
    #batch_size：批大小 
    #True：是否洗牌
    #args:模型参数设置
    #split:可取值为split
    #ALL_SLOTS:所有槽
    model_input = {k:[] for k in pairs[0].keys()}
    #原先使用一个dictionary表示一轮对话，现在拆散成列表，也就是包含字典的列表变为包含列表的字典
    for turn in data:
        for k in model_input.keys():
            model_input[k].append(turn[k]) 
    dataset = Dataset(model_input, global_vocab, mem_vocab, domain_vocab, \
        slot_vocab, args, split, ALL_SLOTS)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,\
                                            batch_size=batch_size,\
                                            shuffle=shuffle,\
                                            collate_fn=collate_fn\
                                        )
    return data_loader

def preprocess_data(args:dict)->None:
    #生成文件存储路径
    if not os.path.exists(args['path']):
        os.makedirs(args['path'])
    #数据加载
    train,dev,test,ontology=None,None,None,None
    try:
        ontology = json.load(open(args['ontology_path'],'r'))
        train=json.load(open(args['train_path'],'r'))
        dev=json.load(open(args['dev_path'],'r'))
        test=json.load(open(args['test_path'],'r'))
    except:
        print("an error occured during reading data(train,dev,test,ontology) from file to RAM!")
        pdb.set_trace()
    #读取ontology中的领域和槽信息
    ALL_SLOTS,ALL_DOMAINS=get_slots_information(ontology)
    global_vocab, mem_vocab, domain_vocab, slot_vocab= Vocabs(), Vocabs(), Vocabs(), Vocabs()
    global_vocab.index_words(ALL_SLOTS,'slot'); global_vocab.index_word('dontcare')
    mem_vocab=deepcopy(global_vocab)
    domain_vocab.index_words(ALL_SLOTS,'domain_only')
    slot_vocab.index_words(ALL_SLOTS,'slot_only')
    if not args['sep_input_embedding']:#将领域\槽加入全局词典
        global_vocab.index_words(domain_vocab.word2index, 'domain_w2i')
        global_vocab.index_words(slot_vocab.word2index, 'slot_w2i')
    #对对话数据进行信息提取
    #特别注意的是,下列函数执行过程中global_vocab, mem_vocab被更新 
    train_data, max_len_a_domainSlotValue_train, max_len_domainSlotValue_train=\
        extract_dialogues(train, ALL_SLOTS, global_vocab, mem_vocab, True,args)
    dev_data, max_len_a_domainSlotValue_dev, max_len_domainSlotValue_dev=\
        extract_dialogues(dev, ALL_SLOTS, global_vocab, mem_vocab, False,args)
    test_data, max_len_a_domainSlotValue_test, max_len_domainSlotValue_test=\
        extract_dialogues(test, ALL_SLOTS, global_vocab, mem_vocab, False,args)
    print("训练数据train包含: %s 次对话" % len(train_data))
    print("验证数据dev包含: %s 次对话" % len(dev_data))
    print("测试数据test包含: %s 次对话" % len(test_data))  
    print("全局词典包含词数: %s " % global_vocab.n_words)
    print("对话状态的词典包含的词数: %s" % mem_vocab.n_words )
    print("领域词典包含的词数: {}".format(domain_vocab.n_words))
    print("槽词典包含的词数: {}".format(slot_vocab.n_words))
    print("槽值最大长度: train {} dev {} test {} all {} ".\
        format(max_len_a_domainSlotValue_train, max_len_a_domainSlotValue_dev, max_len_a_domainSlotValue_test,max(max_len_a_domainSlotValue_train, max_len_a_domainSlotValue_dev, max_len_a_domainSlotValue_test)))
    print("是否使用GPU-USE_CUDA={}".format(USE_CUDA))
    # train_final = get_model_input(train_data, global_vocab, mem_vocab, domain_vocab, slot_vocab, batch_size, True, args, 'train', ALL_SLOTS)
    # dev_final   = get_model_input(dev_data, vocab, mem_vocab, domain_vocab, slot_vocab, eval_batch, False, args, 'dev', ALL_SLOTS)
    # test_final  = get_model_input(test_data, vocab, mem_vocab, domain_vocab, slot_vocab, eval_batch, False, args, 'test', ALL_SLOTS)
    # #存储文件到指定路径
    
    # SLOTS_LIST = {}
    # SLOTS_LIST['all'] = ALL_SLOTS
    # SLOTS_LIST['train'] = slot_train
    # SLOTS_LIST['dev'] = slot_dev
    # SLOTS_LIST['test'] = slot_test
    # return train, dev, test, lang, mem_lang, domain_lang, slot_lang, SLOTS_LIST, max_len_val
    # train, dev, test, \
    # src_lang, tgt_lang, \
    # domain_lang, slot_lang,\
    # SLOTS_LIST, max_len_val = preprocess_multiWOZ(True,args)
    # save_data = {
    #     'train': train, 'dev': dev, 'test': test,
    #     'src_lang': src_lang, 'tgt_lang': tgt_lang,
    #     'domain_lang': domain_lang, 'slot_lang': slot_lang,
    #     'SLOTS_LIST': SLOTS_LIST,
    #     'args': args
    # }
    # if not os.path.exists(args['path']):
    #     os.makedirs(args['path'])
    # joblib.dump(save_data, filename=args['path'] + '/processedMultiWOZ.joblib')
    # print('Storing data into specific path was accomplished!')
    #joblib.dump(save_data, filename=args['path'] + '/processedMultiWOZ.joblib')

if __name__=='__main__':
    preprocess_data(args)