#-*- encoding:utf-8 -*-
from a_config import PAD_token,SOS_token,EOS_token,UNK_token,GATES,REVERSE_GATES

class Lang:
    def __init__(self):
        self.word2index = {}
        #代码写在config.py
        # PAD_token = 1
        # SOS_token = 3 #Start of Sentence
        # EOS_token = 2 #End of Sentence 
        # UNK_token = 0 #unknown token
        self.index2word = {UNK_token: 'UNK', PAD_token: "PAD", EOS_token: "EOS", SOS_token: "SOS"}
        self.n_words = len(self.index2word) # Count default tokens
        self.word2index = dict([(v, k) for k, v in self.index2word.items()])
    
    def index_words(self, sent, type):
        if type == 'utter':#将用户话语和机器话语加入词典
            for word in sent.split():
                self.index_word(word)
        elif type == 'slot':#将domain-slot names加入词典
            for slot in sent:
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split():
                    self.index_word(ss)
        elif type == 'domain':#将domain加入词典
            for domain in sent:
                self.index_word(domain)
        elif type == 'word':#将单词加入词典
            for w in sent:
                self.index_word(w)
        elif type == 'belief':#将信念状态加入词典
            for slot, value in sent.items():
                d, s = slot.split("-")
                self.index_word(d)
                for ss in s.split():
                    self.index_word(ss)
                for v in value.split():
                    self.index_word(v)
        elif type == 'domain_only':#为领域名加上标签加入词典
            for slot in sent:
                d,s = slot.split('-')
                self.index_word('{}_DOMAIN'.format(d))
        elif type == 'slot_only':#为槽加上标签加入词典
            for slot in sent:
                d,s = slot.split('-')
                self.index_word('{}_SLOT'.format(s))
        elif type == 'domain_tag':
            for slot in sent:
                d,s = slot.split('-')
                self.index_word(d)
        elif type == 'slot_tag':
            for slot in sent:
                d,s = slot.split('-')
                self.index_word(s)
        elif type == 'domain_w2i':
            for k,v in sent.items():
                if v<4: continue
                self.index_word('{}_DOMAIN'.format(k))
        elif type == 'slot_w2i':
            for k,v in sent.items():
                if v<4: continue
                self.index_word('{}_SLOT'.format(k))

    def index_word(self, word):#加入word到词典
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1


