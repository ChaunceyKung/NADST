#-*- encoding: utf-8 -*-
#public packages
import torch
import torch.utils.data as data
#user packages
from utils.parameter_setting import USE_CUDA 

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data_info, global_vocab, mem_vocab, domain_vocab, slot_vocab, args, split, ALL_SLOTS):
        """Reads source and target sequences from txt files."""
        self.src_word2id = global_vocab.word2index
        self.mem_word2id = mem_vocab.word2index
        self.domain_word2id = domain_vocab.word2index
        self.slot_word2id = slot_vocab.word2index
        self.all_slots = ALL_SLOTS
        self.args = args
        self.split = split
        for key in data_info.keys():
            if key in {'domainslots_index_auto','y_val_auto'} and not args['auto_regressive']:
                continue
            exec('self.{}=data_info[\'{}\']'.format(key,key))
        self.num_total_seqs = len(self.dialog_history)
    
    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        for key in {"ID","turn_id","dialog_history","delex_dialog_history","turn_uttr",\
            "ordered_domainSlots",'ordered_domains','ordered_slots',"turn_belief_dict",\
            "turn_belief_list",'normalized_val_len','gates','y_val_seq','domains_seq',\
            'slots_seq','domainslots_index_auto','y_val_auto'}:
            if not self.args['auto_regressive'] and \
                key in {'domainslots_index_auto','y_val_auto'}:
                continue
            exec('{}=self.{}[index]'.format(key,key))
        #domain-slots
        if not self.args['sep_input_embedding']:
            ordered_domains_idx = self.preprocess_seq(ordered_domains, self.src_word2id)
            ordered_slots_idx = self.preprocess_seq(ordered_slots, self.src_word2id)
        else:
            ordered_domains_idx = self.preprocess_seq(ordered_domains, self.domain_word2id)
            ordered_slots_idx = self.preprocess_seq(ordered_slots, self.slot_word2id)
        #dialogue history
        context_idx = self.preprocess(dialog_history, self.src_word2id)
        delex_context = None
        if self.args['delex_his']:
            temp = delex_dialog_history.split()
            delex_context = ' '.join(temp)
            delex_context_idx = self.preprocess(delex_context, self.src_word2id)
        #nonauto regressive
        if not self.args['sep_input_embedding']:
            domains_seq_idx = self.preprocess_seq(domains_seq, self.src_word2id)
            slots_seq_idx = self.preprocess_seq(slots_Seq, self.src_word2id)
        else:
            domains_seq_idx = self.preprocess_seq(domains_seq, self.domain_word2id)
            slots_seq_idx = self.preprocess_seq(slots_Seq, self.slot_word2id)
        if self.args['pointer_decoder']:
            y_val_seq_idx = self.preprocess_seq(y_val_seq, self.src_word2id)
        else:
            y_val_seq_idx = self.preprocess_seq(y_val_seq, self.mem_word2id)
        gates = None
        if self.sorted_gates[index] != None:
            gates = gates
        #auto regressive
        auto_ds_idx, auto_y_in, auto_y_out = None, None, None
        if self.args['auto_regressive']:
            auto_ds_idx = domainslots_index_auto
            auto_y_in, auto_y_out = self.preprocess_atrg_seq(y_val_auto, self.src_word2id)
        item_info = {
            "ID":ID,
            "turn_id":turn_id,
            "turn_belief_list":turn_belief_list,
            #context
            "context_plain":dialog_history,
            "context_idx":context_idx,
            "delex_context_plain": delex_dialog_history,
            "delex_context_idx": delex_context_idx,
            #domains slots
            "ordered_domains_idx": ordered_domains_idx,
            "ordered_slots_idx": ordered_slots_idx,
            #non-auto regressive
            "domains_seq_idx": domains_seq_idx,
            "slots_seq_idx": slots_seq_idx,
            "normalized_val_len": normalized_val_len,
            "y_val_seq_idx": y_val_seq_idx,
            "gates": gates,
            #autor regressive
            "auto_ds_idx": auto_ds_idx,
            "auto_y_in": auto_y_in,
            "auto_y_out": auto_y_out
        }
        return item_info
    
    def __len__(self):
        return self.num_total_seqs

    def preprocess(self, sequence, word2idx):
        """Converts words to ids."""
        story = [word2idx[word] if word in word2idx else UNK_token for word in sequence.split()]
        story = torch.Tensor(story)
        return story

    def preprocess_seq(self, sequence, word2idx):
        """Converts words to ids."""
        story = []
        for value in sequence:
            #v = [word2idx[word] if word in word2idx else UNK_token for word in value.split()] + [EOS_token]
            story.append(word2idx[value] if value in word2idx else UNK_token)
        story = torch.Tensor(story)
        return story

    def preprocess_atrg_seq(self, seqs, word2idx):
        y_in = []
        y_out = []
        for seq in seqs:
            seq = ['SOS'] + seq + ['EOS']
            seq = self.preprocess_seq(seq, word2idx)
            y_in.append(seq[:-1])
            y_out.append(seq[1:])
        return y_in, y_out

def collate_fn(data):
    def merge(sequences, pad_token, max_len=-1):
        '''
        merge from batch * sent_len to batch * max_len 
        '''
        lengths = [len(seq) for seq in sequences]
        if max_len < 0:
            max_len = 1 if max(lengths)==0 else max(lengths)
        else:
            assert max_len >= max(lengths)
        padded_seqs = torch.ones(len(sequences), max_len).long() * pad_token
        for i, seq in enumerate(sequences):
            end = lengths[i]
            if type(seq) == list:
                padded_seqs[i, :end] = torch.Tensor(seq[:end])
            else:
                padded_seqs[i, :end] = seq[:end]
        #缃戠粶姊害鍙嶅悜浼犳挱鍦╠etach鍚庣殑tensor鏃犳硶缁х画鍚戝墠浼�
        padded_seqs = padded_seqs.detach() #torch.tensor(padded_seqs)
        return padded_seqs, lengths

    def merge_2d(seqs, pad_token):
        #batch*domain_slots*L_i , L_i涓虹i涓爣绛惧惈鏈夌殑鍗曡瘝鏁�
        temp = []
        for seq in seqs:
            temp += seq
        return merge(temp, pad_token)

    def get_mask(seqs, pad_token):
        #unsqueeze(-2): batch_size*max_len->batch_size*1*max_len, 澧炲姞浜嗕竴缁�
        return (seqs != pad_token).unsqueeze(-2)

    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

    def subsequent_mask(size):
        "Mask out subsequent positions."
        attn_shape = (1, size, size)
        subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
        return torch.from_numpy(subsequent_mask) == 0

    # sort a list by sequence length (descending order) to use pack_padded_sequence
    data.sort(key=lambda x: len(x['context_idx']), reverse=True)
    item_info = {}
    for key in data[0].keys():
        item_info[key] = [d[key] for d in data]
    # context
    context, _ = merge(item_info['context_idx'], PAD_token)
    context_seqs_mask = get_mask(context_seqs, PAD_token)
    delex_context, delex_context_mask = None, None
    if item_info['delex_context_idx'][0] != None:
        delex_context, _ = merge(item_info['delex_context_idx'], PAD_token)
        delex_context_mask = get_mask(delex_context, PAD_token)
    #domain-slots
    ordered_domains_seqs, _ = merge(item_info['ordered_domains_idx'], PAD_token)
    ordered_slots_seqs, _ = merge(item_info['ordered_slots_idx'], PAD_token)
    #nonauto-regressive label
    nonauto_domains_seqs, _ = merge(item_info['domains_seq_idx'], PAD_token)
    nonauto_slots_seqs, _ = merge(item_info['slots_seq_idx'], PAD_token)
    nonauto_domains_seqs_mask = get_mask(nonauto_domains_seqs, PAD_token)
    nonauto_normalized_val_len_seqs, _ = merge(item_info['normalized_val_len'], 0)
    nonauto_y_val_seqs, _ = merge(item_info['y_val_seq_idx'], PAD_token)
    nonauto_gates = None
    if item_info['gates'][0] is not None:
        nonauto_gates, _ = merge(item_info['gates'], GATES['none'])
    #auto-regressive label
    auto_y_in, auto_y_out, auto_y_mask, auto_domainslots_index = None, None, None, None
    if item_info['auto_y_in'][0] != None:
        auto_y_in, _ = merge_2d(item_info['auto_y_in'], PAD_token)
        auto_y_out, _ = merge_2d(item_info['auto_y_out'], PAD_token)
        auto_y_mask = make_std_mask(y_in, PAD_token)
        auto_domainslots_index = []
        for idx, i in enumerate(item_info['auto_ds_idx']):
            temp = [ii + idx*ordered_domains_seqs.shape[1] for ii in i]
            auto_domainslots_index += temp
        auto_domainslots_index = torch.tensor(auto_domainslots_index)
    
    if USE_CUDA:
        ordered_domains_seqs = ordered_domains_seqs.cuda()
        ordered_slots_seqs = ordered_slots_seqs.cuda()

        context = context.cuda()
        context_seqs_mask = context_seqs_mask.cuda()
        if item_info['delex_context_ids'][0] != None:
            delex_context = delex_context.cuda()
            delex_context_mask = delex_context_mask.cuda()
        
        nonauto_domains_seqs = nonauto_domains_seqs.cuda()
        nonauto_domains_seqs_mask=nonauto_domains_seqs_mask.cuda()
        nonauto_slots_seqs = nonauto_slots_seqs.cuda()
        nonauto_normalized_val_len_seqs = nonauto_normalized_val_len_seqs.cuda()
        nonauto_y_val_seqs = nonauto_y_val_seqs.cuda()
        if item_info['gates'][0] != None:
            nonauto_gates = nonauto_gates.cuda()
        
        if item_info['auto_y_in'][0] != None:
            auto_domainslots_index = auto_domainslots_index.cuda()
            auto_y_in = auto_y_in.cuda()
            auto_y_out = auto_y_out.cuda()
            auto_y_mask = auto_y_mask.cuda()
    #context
    item_info["context"] = context
    item_info["context_mask"] = context_mask
    item_info["delex_context"] = delex_context
    item_info["delex_context_mask"] = delex_context_mask
    #domains, slots
    item_info['ordered_domains_seqs'] = ordered_domains_seqs
    item_info['ordered_slots_seqs'] = ordered_slots_seqs
    #nonauto regressive label
    item_info['nonauto_domains_seqs'] = nonauto_domains_seqs
    item_info['nonauto_slots_seqs'] = nonauto_slots_seqs
    item_info['nonauto_val_len'] = nonauto_normalized_val_len_seqs
    item_info['nonauto_y_val_seqs'] = nonauto_y_val_seqs
    item_info['nonauto_gates'] = nonauto_gates
    item_info['sorted_in_domainslots_mask'] = nonauto_domains_seqs_mask
    #auto regressive label
    item_info['auto_domainslots_index'] = auto_domainslots_index
    item_info['auto_y_in'] = auto_y_in
    item_info['auto_y_out'] = auto_y_out
    item_info['auto_y_mask'] = auto_y_mask
    return item_info
