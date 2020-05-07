#-*- encoding: utf-8 -*-
#public packages
import json
import copy
import pdb
import torch
import joblib
import os

#user program
from a_config import args,PAD_token,SOS_token,EOS_token,UNK_token,GATES,REVERSE_GATES,USE_CUDA
from utils_lang import Lang
from utils_dataset import Dataset
from utils_dataset import collate_fn

def get_slot_information(ontology):
    #domain-slot_name
    slots = [k.replace(" ","").lower() for k in ontology.keys()]
    all_domains = set([i.split('-')[0] for i in slots])
    return slots, all_domains

def fix_general_label_error(labels, type, slots):
    #处理掉一些标签上的错误
    #labels: turn-labels
    #type: True or False
    #slots: ALL_SLOTS
    #return: label_slots: a dictionary containing pairs of slot:value
    label_dict = {l[0]: l[1] for l in labels} if type else {l["slots"][0][0]:l["slots"][0][1] for l in labels}
    #type->True: label_dict(slot:<list>slot_value,act:<str>act_name); type->False: label_dict(slot:value)
    GENERAL_TYPO = {# type
        "guesthouse":"guest house", "guesthouses":"guest house", "guest":"guest house", "mutiple sports":"multiple sports", 
        "sports":"multiple sports", "mutliple sports":"multiple sports","swimmingpool":"swimming pool", "concerthall":"concert hall", 
        "concert":"concert hall", "pool":"swimming pool", "night club":"nightclub", "mus":"museum", "ol":"architecture", 
        "colleges":"college", "coll":"college", "architectural":"architecture", "musuem":"museum", "churches":"church",
        # area
        "center":"centre", "center of town":"centre", "near city center":"centre", "in the north":"north", "cen":"centre", "east side":"east", 
        "east area":"east", "west part of town":"west", "ce":"centre",  "town center":"centre", "centre of cambridge":"centre", 
        "city center":"centre", "the south":"south", "scentre":"centre", "town centre":"centre", "in town":"centre", "north part of town":"north", 
        "centre of town":"centre", "cb30aq": "none",
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
    for slot in slots:
        if slot in label_dict.keys():
            # 槽值简写恢复
            label_dict[slot] = GENERAL_TYPO.get(label_dict[slot],label_dict[slot])
            # 槽值匹配不上，槽值根本不在ontology范围内，语义上也说不通
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

def fix_book_slot_name(turn_belief_dict, slots):
    out = {}
    for k in turn_belief_dict.keys():
        new_k = k.replace(" ", "")
        if new_k not in slots: pdb.set_trace()
        out[new_k] = turn_belief_dict[k]
    return out

def fix_multival(turn_belief_dict, multival_count):
    has_multival = False
    for k,v in turn_belief_dict.items():
        if '|' in v:
            values = v.split('|')
            turn_belief_dict[k] =  values[0] #' ; '.join(values)
            has_multival = True
    if has_multival: multival_count += 1 
    return turn_belief_dict, multival_count

def get_sorted_lenval(sorted_domainslots, turn_belief_dict, slot_gating):
    #sorted_domainslots(list):a sorted list of domain_slotname
    #turn_belief_dict:turn label
    #slot_gating: 0 default
    #返回槽值的长度，槽的大类类型，例如gen,none,dontcare,指示后方模型是否启用
    sorted_lenval = [0] * len(sorted_domainslots)
    if slot_gating:
        sorted_gates = [GATES['none']] * len(sorted_domainslots)
    else:
        sorted_gates = None
    for k,v in turn_belief_dict.items():
        index = sorted_domainslots.index(k)
        lenval = len(v.split())
        if not slot_gating or (slot_gating and v not in ['dontcare', 'none']):
            sorted_lenval[index] = lenval
        if slot_gating:
            if v not in ['dontcare', 'none']:
                sorted_gates[index] = GATES['gen']
            else:
                sorted_gates[index] = GATES[v]
    return sorted_lenval, sorted_gates

def get_sorted_generate_y(sorted_domainslots, sorted_lenval_ls, turn_belief_dict):
    #sorted_domainslts:a sorted list of domain_slotname
    #sorted_lenval_ls: len(slot_value)
    #turn_belief_dict: turn_label
    in_domains = []
    in_slots = []
    in_domainslots_index = []
    out_vals = []
    for idx, lenval in enumerate(sorted_lenval_ls):
        if lenval == 0: continue 
        domain = sorted_domainslots[idx].split('-')[0]
        slot = sorted_domainslots[idx].split('-')[1]
        val = turn_belief_dict[sorted_domainslots[idx]].split()
        assert len(val) == lenval
        for i in range(lenval):
            in_domains.append(domain+"_DOMAIN")
            in_slots.append(slot+"_SLOT")
            out_vals.append(val[i])
            in_domainslots_index.append(idx)
    return in_domains, in_slots, out_vals, in_domainslots_index

def get_atrg_generate_y(sorted_domainslots, sorted_lenval_ls, turn_belief_dict):
    vals = []
    indices = []
    for idx, lenval in enumerate(sorted_lenval_ls):
        if lenval == 0: continue 
        val = turn_belief_dict[sorted_domainslots[idx]].split()
        assert len(val) == lenval
        vals.append(val)
        indices.append(idx)
    return vals, indices

def read_langs(file_name, ALL_SLOTS, dataset, lang, mem_lang, is_training, args):
    #file_name: corpus file name, such as multiWOZ21.json
    #SLOTS: all domain_slotname read from ontology
    #dataset: a str object, which is a element of {'train','dev','test'}
    #lang: a dictionary containing all slots
    #mem_lang: a similar object to lang
    #training: True or False, indicate the pattern is training or not
    #args: program parameters
    sorted_domainslots = sorted(ALL_SLOTS)
    sorted_in_domains = [i.split('-')[0]+"_DOMAIN" for i in sorted_domainslots]
    sorted_in_slots = [i.split('-')[1]+"_SLOT" for i in sorted_domainslots]
    
    max_len_slot_val = {}#记录domain-slot可取值的最长长度与值tuple(len(value.split(),value)
    domain_counter = {}#统计每个领域在语料中有多少次对话
    data = []
    max_len_val_per_slot = 0
    multival_count = 0#统计槽值可取多值的对话状态数(一轮次对话有一个状态，一次对话有n轮次)
    # counting none/dontcare
    for ds in sorted_domainslots:
        max_len_slot_val[ds] = (1, "none") 
    with open(file_name) as f:
        dials = json.load(f)
        # 将用户话语和系统话语加入词典全局词典lang
        for dial_dict in dials:
            if (dataset=='train' and is_training) or (args['pointer_decoder']):
                for ti, turn in enumerate(dial_dict["dialogue"]):
                    lang.index_words(turn["system_transcript"], 'utter')
                    lang.index_words(turn["transcript"], 'utter')
        
        for dial_dict in dials:
            last_belief_dict = {}#标签
            # 统计每个领域包含的对话数
            for domain in dial_dict["domains"]:
                if domain not in domain_counter.keys():
                    domain_counter[domain] = domain.get(domain,0)+1
            # Reading data
            dialog_history = ''#对话历史
            delex_dialog_history = ''#包含标记的对话历史
            for ti, turn in enumerate(dial_dict["dialogue"]):
                turn_id = turn["turn_idx"]#对话的标识
                if ti == 0:
                    user_sent = ' SOS ' + turn["transcript"] + ' EOS '
                    dlx_user_sent = ' SOS ' + turn["delex_transcript"] + ' EOS '
                    sys_sent = ''
                    dlx_sys_sent = ''
                else:
                    sys_sent = ' SOS '  + turn["system_transcript"] + ' EOS '
                    user_sent = 'SOS ' + turn["transcript"] + ' EOS '
                    dlx_sys_sent = ' SOS '  + turn["delex_system_transcript"] + ' EOS '
                    dlx_user_sent = 'SOS ' + turn["delex_transcript"] + ' EOS '
                #对话状态词典化
                turn_belief_dict = fix_general_label_error(turn["belief_state"], False, ALL_SLOTS)#处理掉一般性的标签错误，返回槽值对词典slot:value
                turn_belief_dict = fix_book_slot_name(turn_belief_dict, ALL_SLOTS)#将上一个语句中的空格删掉
                turn_belief_dict, multival_count = fix_multival(turn_belief_dict, multival_count)#将槽可取多个值的情况删除，只剩下第一个值
                turn_belief_dict = {k:v for k,v in turn_belief_dict.items() if v!='none'}#删除'none'的槽值对
                for k,v in turn_belief_dict.items():
                    if len(v.split()) > max_len_slot_val[k][0]:
                        max_len_slot_val[k] = (len(v.split()), v)
                #对话状态辅助向量： sorted_lenval:指示在sorted_domainslots的序下对应的槽取值的长度，sorted_gates则指示在sorted_domainslots的序下对应的槽取值是否需要生成
                sorted_lenval, sorted_gates = get_sorted_lenval(sorted_domainslots, turn_belief_dict, args['slot_gating'])#获取对应槽值长度
                if len(sorted_lenval)>0 and max(sorted_lenval) > max_len_val_per_slot:
                    max_len_val_per_slot = max(sorted_lenval)
                # turn_belief: {domain-slot:value...}，non-autoregressive
                #domain_slot_index每个非空取值槽的对应sorted_domainslot下标,包含的下标一定有序
                #sorted_generate_y:按sorted_domainslot顺序非空槽取值(一个槽对应的槽值长度大于1，则在generate_y中包含多个值对应该槽)
                #sorted_in_slots:非空槽的槽序列，只含slot
                #sorted_in_domains2:只含领域的领域序列
                sorted_in_domains2, sorted_in_slots2, sorted_generate_y, sorted_in_domainslots2_index = get_sorted_generate_y(sorted_domainslots, sorted_lenval, turn_belief_dict)
                if args['auto_regressive']:
                    #slot_values,domain_slot_index
                    atrg_generate_y, sorted_in_domainslots2_index = get_atrg_generate_y(sorted_domainslots, sorted_lenval, turn_belief_dict)
                else:
                    atrg_generate_y = None 
                if args['delex_his']:
                    in_delex_dialog_history= delex_dialog_history+dlx_sys_sent+user_sent
                    if (dataset=='train' and is_training) or (args['pointer_decoder']):
                        lang.index_words(in_delex_dialog_history, 'utter')
                turn_uttr = sys_sent + user_sent 
                dialog_history += sys_sent +user_sent
                delex_dialog_history += dlx_sys_sent+dlx_user_sent

                turn_belief_list = [str(k)+'-'+str(v) for k, v in turn_belief_dict.items()]
                if dataset=='train' and is_training: 
                    mem_lang.index_words(turn_belief_dict, 'belief')
                data_detail = {#一轮次对话
                    "ID":dial_dict["dialogue_idx"], 
                    "turn_id":turn_id, 
                    "dialog_history": dialog_history.strip(), 
                    "delex_dialog_history": in_delex_dialog_history.strip(),
                    "turn_belief":turn_belief_list,
                    "sorted_domainslots": sorted_domainslots,
                    "turn_belief_dict": turn_belief_dict, 
                    "turn_uttr":turn_uttr.strip(), 
                    'sorted_in_domains': sorted_in_domains,
                    'sorted_in_slots': sorted_in_slots,
                    'sorted_in_domains2': sorted_in_domains2,
                    'sorted_in_slots2': sorted_in_slots2,
                    'sorted_in_domainslots2_idx': sorted_in_domainslots2_index, 
                    'sorted_lenval': sorted_lenval,
                    'sorted_gates': sorted_gates, 
                    'sorted_generate_y': sorted_generate_y,
                    'atrg_generate_y': atrg_generate_y
                    }
                data.append(data_detail)
    print('File name:',file_name)
    print("multival_count: ", multival_count) 
    print("domain_counter: ", domain_counter,end='\n\n')
    
    return data, ALL_SLOTS, max_len_val_per_slot, max_len_slot_val

def get_seq(pairs, lang, mem_lang, domain_lang, slot_lang, batch_size, shuffle, args, split, ALL_SLOTS):  
    #pair: 对话的数据，包含字典的列表
    #lang: 全局字典 mem_lang：对话状态的字典; domain_lang:领域词典 slot_lang：槽词典
    #batch_size：批大小 
    #True：是否洗牌
    #args:模型参数设置
    #split:可取值为split
    #ALL_SLOTS:所有槽
    data_info = {}
    data_keys = pairs[0].keys()
    for k in data_keys:
        data_info[k] = []
    #原先使用一个dictionary表示轮对话，现在拆散成列表，也就是包含字典的列表变为包含列表的字典
    for pair in pairs:
        for k in data_keys:
            data_info[k].append(pair[k]) 
    dataset = Dataset(data_info, lang, mem_lang, domain_lang, slot_lang, args, split, ALL_SLOTS)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  collate_fn=collate_fn)
    return data_loader

def preprocess_multiWOZ(training,args):
    batch_size = args['batch']#32
    eval_batch = args["eval_batch"] if args["eval_batch"] else batch_size#16
    file_train = 'data{}/nadst_train_dials.json'.format(args['data_version'])
    file_dev = 'data{}/nadst_dev_dials.json'.format(args['data_version'])
    file_test = 'data{}/nadst_test_dials.json'.format(args['data_version'])
    ontology = json.load(open("data2.0/multi-woz/MULTIWOZ2 2/ontology.json", 'r'))
    ALL_SLOTS, ALL_DOMAINS = get_slot_information(ontology)
    # def get_slot_information(ontology):
    # #domain-slot_name
    #     slots = [k.replace(" ","").lower() for k in ontology.keys()]
    #     all_domains = set([i.split('-')[0] for i in slots])
    #     return slots, all_domains
    #lang:词典函数
    lang = Lang()#全局词典
    lang.index_words(ALL_SLOTS, 'slot'); lang.index_word('dontcare')
    mem_lang=copy.deepcopy(lang)#信念状态词典
    #domain_lang:领域词典 slot_lang:槽词典
    domain_lang, slot_lang = Lang(), Lang()
    domain_lang.index_words(ALL_SLOTS, 'domain_only')
    slot_lang.index_words(ALL_SLOTS, 'slot_only')

    #构建模型输入数据
    if training:
        pair_train, slot_train, train_max_len_val, train_max_len_slot_val = read_langs(file_train, ALL_SLOTS, "train", lang, mem_lang, training, args)
        pair_dev, slot_dev, dev_max_len_val, dev_max_len_slot_val = read_langs(file_dev, ALL_SLOTS, "dev", lang, mem_lang, training, args)
        pair_test, slot_test, test_max_len_val, test_max_len_slot_val = read_langs(file_test, ALL_SLOTS, "test", lang, mem_lang, training, args)
        max_len_val = max(train_max_len_val, dev_max_len_val, test_max_len_val)
        if not args['sep_input_embedding']:
            lang.index_words(domain_lang.word2index, 'domain_w2i')
            lang.index_words(slot_lang.word2index, 'slot_w2i')
        train = get_seq(pair_train, lang, mem_lang, domain_lang, slot_lang, batch_size, True, args, 'train', ALL_SLOTS)
        dev   = get_seq(pair_dev, lang, mem_lang, domain_lang, slot_lang, eval_batch, False, args, 'dev', ALL_SLOTS)
        test  = get_seq(pair_test, lang, mem_lang, domain_lang, slot_lang, eval_batch, False, args, 'test', ALL_SLOTS)
    print("训练数据train包含: %s 次对话" % len(pair_train))
    print("验证数据dev包含: %s 次对话" % len(pair_dev))
    print("测试数据test包含: %s 次对话" % len(pair_test))  
    print("全局词典包含词数: %s " % lang.n_words)
    print("训练使用的词典包含的词数: %s" % lang.n_words )
    print("对话状态的词典包含的词数: %s" % mem_lang.n_words )
    print("领域词典包含的词数: {}".format(domain_lang.n_words))
    print("槽词典包含的词数: {}".format(slot_lang.n_words))
    print("槽值最大长度: train {} dev {} test {} all {}".format(train_max_len_val, dev_max_len_val, test_max_len_val, max_len_val))
    print("USE_CUDA={}".format(USE_CUDA))
    SLOTS_LIST = {}
    SLOTS_LIST['all'] = ALL_SLOTS
    SLOTS_LIST['train'] = slot_train
    SLOTS_LIST['dev'] = slot_dev
    SLOTS_LIST['test'] = slot_test
    return train, dev, test, lang, mem_lang, domain_lang, slot_lang, SLOTS_LIST, max_len_val

def preprocess_and_storing():
    train, dev, test, \
    src_lang, tgt_lang, \
    domain_lang, slot_lang,\
    SLOTS_LIST, max_len_val = preprocess_multiWOZ(True,args)
    save_data = {
        'train': train, 'dev': dev, 'test': test,
        'src_lang': src_lang, 'tgt_lang': tgt_lang,
        'domain_lang': domain_lang, 'slot_lang': slot_lang,
        'SLOTS_LIST': SLOTS_LIST,
        'args': args
    }
    if not os.path.exists(args['path']):
        os.makedirs(args['path'])
    joblib.dump(save_data, filename=args['path'] + '/processedMultiWOZ.joblib')
    print('Storing data into specific path was accomplished!')
