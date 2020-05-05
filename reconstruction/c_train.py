#-*- encoding:utf-8 -*-
#public packages
import torch.nn as nn
import joblib
import os 

#private source file
from utils_label_smoothing import LabelSmoothing
from model.optimizer import NoamOpt,LossCompute
from model.evaluator import Evaluator
from a_config import args,PAD_token,SOS_token,EOS_token,UNK_token,GATES,REVERSE_GATES,USE_CUDA
from b_preprocess import preprocess_and_storing

def run_epoch(epoch, max_epoch, data, model, is_eval):
    avg_lenval_loss = 0 
    avg_gate_loss = 0
    avg_state_loss = 0
    
    epoch_lenval_loss = 0
    epoch_gate_loss = 0
    epoch_state_loss = 0
    
    avg_slot_nb_tokens = 0
    avg_state_nb_tokens = 0
    avg_gate_nb_tokens = 0
    
    epoch_slot_nb_tokens = 0
    epoch_state_nb_tokens = 0
    epoch_gate_nb_tokens = 0 
    
    epoch_joint_lenval_matches = 0
    epoch_joint_gate_matches = 0 
    total_samples = 0
    
    if args['pointer_decoder']:
        predict_lang = src_lang 
    else:
        predict_lang = tgt_lang
    
    pbar = tqdm(enumerate(data),total=len(data), desc="epoch {}/{}".format(epoch+1, max_epoch), ncols=0)
    if is_eval:
        predictions = {}
    for i, data in pbar:
        out = model.forward(data)
        losses, nb_tokens = loss_compute(out, data, is_eval)
        if is_eval:  #and do_predict:
            matches, predictions = predict(out, data, model, predict_lang, domain_lang, slot_lang, predictions, True, src_lang, args)
            epoch_joint_lenval_matches += matches['joint_lenval']
            epoch_joint_gate_matches += matches['joint_gate']
            total_samples += len(data['turn_id'])
      
        avg_lenval_loss += losses['lenval_loss']
        avg_gate_loss += losses['gate_loss']
        avg_state_loss += losses['state_loss']

        avg_gate_nb_tokens += nb_tokens['gate']
        avg_slot_nb_tokens += nb_tokens['slot']
        avg_state_nb_tokens += nb_tokens['state']
        
        epoch_slot_nb_tokens += nb_tokens['slot']
        epoch_state_nb_tokens += nb_tokens['state']
        epoch_gate_nb_tokens += nb_tokens['gate']

        epoch_lenval_loss += losses['lenval_loss']
        epoch_state_loss += losses['state_loss']
        epoch_gate_loss += losses['gate_loss']
        
        if (i+1) % args['reportp'] == 0 and not is_eval: 
            avg_lenval_loss /= avg_slot_nb_tokens 
            avg_state_loss /= avg_state_nb_tokens
            avg_gate_loss /= avg_gate_nb_tokens
            print("Step {} gate loss {:.4f} lenval loss {:.4f} state loss {:.4f}".
                format(i+1, avg_gate_loss, avg_lenval_loss, avg_state_loss))
            with open(args['path'] + '/train_log.csv', 'a') as f:
                f.write('{},{},{},{},{}\n'.format(epoch+1, i+1, avg_gate_loss, avg_lenval_loss, avg_state_loss))
            avg_lenval_loss = 0
            avg_slot_nb_tokens = 0 
            avg_state_loss = 0
            avg_state_nb_tokens = 0
            avg_gate_loss = 0
            avg_gate_nb_tokens = 0

    epoch_lenval_loss /= epoch_slot_nb_tokens 
    epoch_state_loss /= epoch_state_nb_tokens
    epoch_gate_loss /= epoch_gate_nb_tokens 
    joint_gate_acc, joint_lenval_acc, joint_acc_score, F1_score, turn_acc_score = 0, 0, 0, 0, 0
    
    real_joint_acc_score = 0.0
    if is_eval:
        joint_lenval_acc = 1.0 * epoch_joint_lenval_matches/total_samples
        joint_gate_acc = 1.0 * epoch_joint_gate_matches/total_samples 
        joint_acc_score, F1_score, turn_acc_score = -1, -1, -1
        joint_acc_score, F1_score, turn_acc_score = evaluator.evaluate_metrics(predictions, 'dev')
            
    print("Epoch {} gate loss {:.4f} lenval loss {:.4f} state loss {:.4f} \n joint_gate acc {:.4f} joint_lenval acc {:.4f} joint acc {:.4f} f1 {:.4f} turn acc {:.4f}".
        format(epoch+1, epoch_gate_loss, epoch_lenval_loss, epoch_state_loss, 
               joint_gate_acc, joint_lenval_acc, joint_acc_score, F1_score, turn_acc_score))
    print(args['path'])
    with open(args['path'] + '/val_log.csv', 'a') as f:
        if is_eval: 
            split='dev'
        else:
            split='train'
        f.write('{},{},{},{},{},{},{},{},{},{}\n'.
            format(epoch+1,split,
                   epoch_gate_loss, epoch_lenval_loss,epoch_state_loss,
                   joint_gate_acc, joint_lenval_acc,
                   joint_acc_score,F1_score,turn_acc_score))

    return (epoch_gate_loss + epoch_lenval_loss + epoch_state_loss)/3, (joint_gate_acc + joint_lenval_acc + joint_acc_score)/3, joint_acc_score


if __name__=='__main__':
    if not os.path.exists(args['path']+ '/processedMultiWOZ.joblib'):
        preprocess_and_storing()
    save_data=joblib.load(args['path'] + '/processedMultiWOZ.joblib')
    train, dev, test, \
    src_lang, tgt_lang, \
    domain_lang, slot_lang,\
    max_len_val,args= save_data['train'], save_data['dev'], save_data['test'],\
        save_data['src_lang'], save_data['tgt_lang'],\
        save_data['domain_lang'], save_data['slot_lang'],\
        save_data['SLOTS_LIST'],\
        save_data['args']
    for i in train:
        print(i.keys())
        break
    exit()
    model = make_model(
        src_lang = src_lang, tgt_lang = tgt_lang,
        domain_lang = domain_lang, slot_lang = slot_lang, 
        len_val=max_len_val, args=args)
    model.cuda()

    len_val_criterion = nn.CrossEntropyLoss()
    gate_gen_criterion = nn.CrossEntropyLoss()
    if args['pointer_decoder']:
        state_gen_criterion = LabelSmoothing(size=len(src_lang.word2index), padding_idx=PAD_token, smoothing=0.1, run_softmax=False) 
    else:
        state_gen_criterion = LabelSmoothing(size=len(tgt_lang.word2index), padding_idx=PAD_token, smoothing=0.1)      
    opt = NoamOpt(args['d_model'], 1, args['warmup'], torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)) 
    loss_compute = LossCompute(model, len_val_criterion, state_gen_criterion, gate_gen_criterion, opt, args)
    evaluator = Evaluator(SLOTS_LIST)

    with open(args['path'] + '/train_log.csv', 'w') as f:
        f.write('epoch,step,gate_loss,lenval_loss,state_loss\n')
    with open(args['path'] + '/val_log.csv', 'w') as f:
        f.write('epoch,split,gate_loss,lenval_loss,state_loss,joint_gate_acc,joint_lenval_acc,joint_acc,f1,turn_acc\n')
    json.dump(args, open(args['path'] + '/params.json', 'w'))
        
    best_modelfile = args['path'] + '/model_best.pth.tar'
    for epoch in range(200):
        print("Epoch:{}".format(epoch))  
        model.train()
        run_epoch(epoch, 200, train, model, False)
        modelfile = args['path'] + '/model_epoch{}.pth.tar'.format(epoch+1)
        if((epoch+1) % int(args['evalp']) == 0):
            model.eval()
            dev_loss, dev_acc, dev_joint_acc = run_epoch(epoch, -1, dev, model, True) 
            if args['eval_metric'] == 'acc':
                check = (dev_acc > max_dev_acc)
            elif args['eval_metric'] == 'slot_acc':
                check = (dev_joint_acc > max_dev_slot_acc)
            elif args['eval_metric'] == 'loss':
                check = (dev_loss < min_dev_loss)
            if check:
                torch.save(model, modelfile) 
                cnt = 0 
                best_model_id = epoch+1
                print('Dev loss changes from {} --> {}'.format(min_dev_loss, dev_loss))
                print('Dev acc changes from {} --> {}'.format(max_dev_acc, dev_acc))
                print('Dev slot acc changes from {} --> {}'.format(max_dev_slot_acc, dev_joint_acc))
                min_dev_loss = dev_loss
                max_dev_acc = dev_acc
                max_dev_slot_acc = dev_joint_acc
                if os.path.exists(best_modelfile):
                    os.remove(best_modelfile)
                os.symlink(os.path.basename('model_epoch{}.pth.tar'.format(epoch+1)), best_modelfile)
                print('A symbolic link is made as {}'.format(best_modelfile))
            else:
                cnt += 1 
            if(cnt == args["patience"]):
                print("Ran out of patient, early stop...")  
                break 
    print("The best model is at epoch {}".format(best_model_id))
