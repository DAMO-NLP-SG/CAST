import argparse
import os
import json
import numpy as np
import torch
import math
import torch.nn.functional as F
#from memory_profiler import profile
from apex import amp
import ujson as json
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model import DocREModel
from utils import set_seed, collate_fn
from prepro import read_docred
from evaluation import to_official, to_official_by_doc, official_evaluate, official_evaluate_benchmark, score_predictions_by_class
#import wandb
from copy import deepcopy
from sklearn.model_selection import KFold

rel2id = {'P1376': 79, 'P607': 27, 'P136': 73, 'P137': 63, 'P131': 2, 'P527': 11, 'P1412': 38, 'P206': 33, 'P205': 77, 'P449': 52, 'P127': 34, 'P123': 49, 'P86': 66, 'P840': 85, 'P355': 72, 'P737': 93, 'P740': 84, 'P190': 94, 'P576': 71, 'P749': 68, 'P112': 65, 'P118': 40, 'P17': 1, 'P19': 14, 'P3373': 19, 'P6': 42, 'P276': 44, 'P1001': 24, 'P580': 62, 'P582': 83, 'P585': 64, 'P463': 18, 'P676': 87, 'P674': 46, 'P264': 10, 'P108': 43, 'P102': 17, 'P25': 81, 'P27': 3, 'P26': 26, 'P20': 37, 'P22': 30, 'Na': 0, 'P807': 95, 'P800': 51, 'P279': 78, 'P1336': 88, 'P577': 5, 'P570': 8, 'P571': 15, 'P178': 36, 'P179': 55, 'P272': 75, 'P170': 35, 'P171': 80, 'P172': 76, 'P175': 6, 'P176': 67, 'P39': 91, 'P30': 21, 'P31': 60, 'P36': 70, 'P37': 58, 'P35': 54, 'P400': 31, 'P403': 61, 'P361': 12, 'P364': 74, 'P569': 7, 'P710': 41, 'P1344': 32, 'P488': 82, 'P241': 59, 'P162': 57, 'P161': 9, 'P166': 47, 'P40': 20, 'P1441': 23, 'P156': 45, 'P155': 39, 'P150': 4, 'P551': 90, 'P706': 56, 'P159': 29, 'P495': 13, 'P58': 53, 'P194': 48, 'P54': 16, 'P57': 28, 'P50': 22, 'P1366': 86, 'P1365': 92, 'P937': 69, 'P140': 50, 'P69': 25, 'P1198': 96, 'P1056': 89}


crest_probs = [(1, 0.08038683629259107), (2, 0.08038683629259107), (3, 0.09896780440809584), (4, 0.12622596251522658), (5, 0.15540238856970603), (6, 0.16726084595862234), (7, 0.1863932239419447), (8, 0.18971267663544578), (11, 0.19132278250804943), (9, 0.19132278250804943), (12, 0.19132278250804943), (10, 0.2085683038889232), (13, 0.21239632155218935), (14, 0.2252129944500172), (15, 0.22629663860684987), (18, 0.22947697634147313), (17, 0.2374900692388342), (16, 0.238448347600796), (20, 0.23939772328571077), (21, 0.24033839452958466), (19, 0.24219438236430132), (22, 0.24219438236430132), (25, 0.24219438236430132), (31, 0.24491765575393426), (26, 0.24580989761034638), (23, 0.2475720517577266), (24, 0.2475720517577266), (27, 0.2535200351480276), (30, 0.2535200351480276), (29, 0.2559723288961981), (28, 0.2559723288961981), (36, 0.25677770213367257), (35, 0.25994169001865747), (32, 0.26225701006254193), (42, 0.2630182839938371), (34, 0.26601312483389805), (37, 0.2674814961428515), (43, 0.2682086818963138), (33, 0.27386720314329693), (45, 0.27386720314329693), (41, 0.2856880721082117), (39, 0.2875506468533753), (40, 0.28999111099831637), (47, 0.28999111099831637), (44, 0.2935643510200467), (49, 0.294733169590695), (48, 0.296466378078828), (46, 0.2970388929975608), (53, 0.30097624493417763), (38, 0.30262748218786384), (52, 0.3058683020804574), (51, 0.3058683020804574), (55, 0.3064007112749972), (50, 0.3126277008521072), (54, 0.31414003874668983), (56, 0.3156355765476086), (58, 0.31613043345341596), (57, 0.31711476042344844), (73, 0.3180920094035695), (62, 0.3214583783925441), (59, 0.32381349094657913), (69, 0.32474444181249584), (60, 0.3306491397243365), (65, 0.3341639009650469), (64, 0.3371700771482673), (61, 0.34053085617943013), (63, 0.3478220546734855), (68, 0.3513376825262286), (72, 0.3521078815111467), (70, 0.3606955774664244), (77, 0.3610582681497857), (67, 0.36250059749694763), (75, 0.3628590951690239), (76, 0.36709803766532534), (71, 0.36848594863207457), (66, 0.3735849569629345), (78, 0.3804616942772294), (79, 0.3817391379332982), (80, 0.387674915751454), (81, 0.3957616866808968), (74, 0.398085208092749), (82, 0.41484324339643286), (84, 0.42403549144589886), (83, 0.4308762613210507), (85, 0.4411400668012653), (87, 0.4440683451160069), (89, 0.44957630981025065), (86, 0.45195069822154693), (90, 0.4859758659525261), (88, 0.525395184305148), (91, 0.5265997639964111), (92, 0.5397290195576788), (93, 0.6389181909949321), (94, 0.6978356927101282), (96, 0.797321356405658), (95, 1.0)]

def train(args, model, config, train_features, dev_features, test_features):
    def finetune(features, optimizer, num_epoch, num_steps):
        best_score = -1
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
        train_iterator = range(int(num_epoch))
        total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
        warmup_steps = int(total_steps * args.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        print("Total steps: {}".format(total_steps))
        print("Warmup steps: {}".format(warmup_steps))
        for epoch in train_iterator:
            model.zero_grad()
            if epoch == -100: 
                tmp_features = find_hard_negatives(args, model, features, tag="distant", neg_rate=0.5, probs=1.0, inference_th=1.0)
                train_dataloader = DataLoader(tmp_features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
            for step, batch in enumerate(train_dataloader):
                model.train()
                inputs = {'input_ids': batch[0].to(args.device),
                          'attention_mask': batch[1].to(args.device),
                          'labels': batch[2],
                          'entity_pos': batch[3],
                          'hts': batch[4],
                          }
                #with torch.autograd.set_detect_anomaly(True):
                outputs = model(**inputs)[0]
                loss = outputs / args.gradient_accumulation_steps
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    #with torch.autograd.set_detect_anomaly(True):
                    scaled_loss.backward()
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                #wandb.log({"loss": loss.item()}, step=num_steps)
                if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0) and (epoch>=20):
                    dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
                    #wandb.log(dev_output, step=num_steps)
                    print(dev_output)
                    if dev_score > best_score:
                        best_score = dev_score
                        pred = report(args, model, test_features)
                        with open("result.json", "w") as fh:
                            json.dump(pred, fh)
                        if args.save_path != "":
                            torch.save(model.state_dict(), args.save_path)
        return model
    '''
    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": args.classifier_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    '''
    model, optimizer = initialize_model_and_optimizer(args, config)
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    finetune(train_features, optimizer, args.num_train_epochs, num_steps)

def finetune_func(args, model, train_features, dev_features, test_features, optimizer, num_epoch, num_steps):
    best_score = -1
    train_dataloader = DataLoader(train_features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    train_iterator = range(int(num_epoch))
    total_steps = int(len(train_dataloader) * num_epoch // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    print("Total steps: {}".format(total_steps))
    print("Warmup steps: {}".format(warmup_steps))
    for epoch in train_iterator:
        model.zero_grad()
        for step, batch in enumerate(train_dataloader):
            model.train()
            inputs = {'input_ids': batch[0].to(args.device),
                      'attention_mask': batch[1].to(args.device),
                      'labels': batch[2],
                      'entity_pos': batch[3],
                      'hts': batch[4],
                        }
            #with torch.autograd.set_detect_anomaly(True):
            outputs = model(**inputs)[0]
            loss = outputs / args.gradient_accumulation_steps
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                #with torch.autograd.set_detect_anomaly(True):
                scaled_loss.backward()
            if step % args.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                num_steps += 1
            #wandb.log({"loss": loss.item()}, step=num_steps)
            if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0) and (epoch>=20):
                dev_score, dev_output = evaluate(args, model, dev_features, tag="dev", inference_th=args.inference_th)
                #wandb.log(dev_output, step=num_steps)
                print(dev_output)
                if dev_score > best_score:
                    best_score = dev_score
                    best_model = deepcopy(model)
                    pred = report(args, model, test_features, inference_th=args.inference_th)
                    with open("result.json", "w") as fh:
                        json.dump(pred, fh)
                    if args.save_path != "":
                        torch.save(model.state_dict(), args.save_path)
    return best_model
    
    
def train_self_training(args, model, config, train_features, distant_features, dev_features, test_features):


    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": args.classifier_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    num_steps = 0
    set_seed(args)
    model.zero_grad()
    model = finetune_func(args, model, train_features, dev_features, test_features, optimizer, args.num_train_epochs, num_steps)
    chunk_size = args.chunk_size
    num_splits = int(math.ceil(len(distant_features)/chunk_size))
    temp_epoch = args.iteration_epoch
    for i in range(num_splits):
        print('Pseudo-label Round: {}'.format(i + 1))
        temp_distant = generate_mixup_label(args, model, distant_features[i * chunk_size: (i + 1) * chunk_size], 
                                            tag="distant", w_self= args.w_self, w_distant = args.w_distant, inference_th = args.inference_th, )
        if args.re_init == True:
            model = DocREModel(config, args.model_name_or_path, args.cache_dir, num_labels=args.num_labels).to(0)
            new_layer = ["extractor", "bilinear"]
            optimizer_grouped_parameters = [
                {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
                {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": args.classifier_lr},
                ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
        model = finetune_func(args, model, temp_distant, dev_features, test_features, optimizer, temp_epoch, num_steps=0)
        model = finetune_func(args, model, train_features, dev_features, test_features, optimizer, temp_epoch, num_steps)
        torch.save(model.state_dict(), 'checkpoints/{}_gt_size-{}_iter-{}_re_init-{}_w_self-{}_w_dist-{}_.pt'.format(args.model_name_or_path, len(train_features), i, args.re_init, args.w_self, args.w_distant ))
        
def get_train_dev_split(a, n):
    k, m = divmod(len(a), n)
    devs = [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]
    trains = [a[:i*k+min(i, m)] + a[(i+1)*k+min(i+1, m):] for i in range(n)]
    return trains, devs

def get_train_split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

def train_KF_self_training(args, config, train_features, distant_features, dev_features, test_features, n_splits):

    global crest_probs 
    splits = n_splits 
    trains, devs = get_train_dev_split(train_features, args.n_splits)
    merged_train = []
    dev_file = os.path.join(args.data_dir, args.dev_file)
    dev_data = json.load(open(dev_file))
    for i in range(splits):
        model, optimizer = initialize_model_and_optimizer(args, config)
        num_steps = 0
        set_seed(args)
        model.zero_grad()
        trains[i] = get_random_mask(trains[i], args.drop_prob, args.device)
        model = finetune_func(args, model, trains[i], dev_features, test_features, optimizer, args.num_train_epochs, num_steps)
        scores_by_class = evaluate_by_class(args, model, dev_features, dev_data, tag="dev", output_logits=False, inference_th = args.inference_th)
        if args.crest == True:
            probs = calculate_probability_crest(crest_probs)
        else:
            probs = calculate_probability(scores_by_class, gamma_p= args.gamma_p, gamma_r = args.gamma_r)
        #devs[i] = generate_and_filter_label(args, model, devs[i], tag="train", w_p = args.w_self, w_d = args.w_distant, d_drop_ratio = args.d_drop_ratio, p_drop_ratio = args.p_drop_ratio, inference_th = args.inference_th)
        devs[i] = generate_class_balance_label(args, model, devs[i], tag="train", w_p = args.w_self, w_d = args.w_distant, d_drop_ratio = args.d_drop_ratio, p_drop_ratio = args.p_drop_ratio, probs = probs, inference_th = args.inference_th)
        
        #torch.save(model.state_dict(), '{}-split-{}-w_self-{}-w_distant-{}.pt'.format(args.save_path[:-3],  i, args.w_self, args.w_distant ))
        del model, optimizer
    pseudo_features = sum(devs, [])
    
    

    return pseudo_features

def initialize_model_and_optimizer(args, config):
    model = DocREModel(config, model=None, model_name=args.model_name_or_path, cache_dir=args.cache_dir, num_labels=args.num_labels).to(args.device)
    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": args.classifier_lr},
            ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    return model, optimizer


def Multi_Round_KF_self_training(args, model, config, train_features, distant_features, dev_features, test_features, n_splits, n_rounds):
    for i in range(args.start_round, args.start_round + n_rounds):
        train_features = train_KF_self_training(args, config, train_features, distant_features, dev_features, test_features, n_splits)
        torch.save(train_features, '{}-round-{}-drop-{}-final-{}-rounds-{}-splits-w_self-{}-w_distant-{}-d_drop_ratio-{}-p_drop_ratio-{}-gamma_p-{}-gamma_r-{}.features'.format(args.save_path[:-3],  i, args.drop_prob, args.n_rounds, args.n_splits, args.w_self, args.w_distant, args.d_drop_ratio, args.p_drop_ratio, args.gamma_p, args.gamma_r ))
        model, optimizer = initialize_model_and_optimizer(args, config)
        num_steps = 0
        set_seed(args)
        model.zero_grad()
        sampled_train_features = get_random_mask(train_features, args.drop_prob, args.device)
        print('Final training for Round: {} Neg Sample rate: {}'.format(i+1, 1 - args.drop_prob))
        model = finetune_func(args, model, sampled_train_features, dev_features, test_features, optimizer, args.num_train_epochs, num_steps)
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev", inference_th = args.inference_th)
        torch.save(model.state_dict(),'{}-round-{}-drop-{}-final-{}-rounds-{}-splits-w_self-{}-w_distant-{}-d_drop_ratio-{}-p_drop_ratio-{}-gamma_p-{}-gamma_r-{}.pt'.format(args.save_path[:-3],  i, args.drop_prob, args.n_rounds, args.n_splits, args.w_self, args.w_distant, args.d_drop_ratio, args.p_drop_ratio, args.gamma_p, args.gamma_r ))
        del model, optimizer
        print('KFold Self Training Round {} Result:'.format(i+1))
        print(dev_output)
    return None
        
        
        
def evaluate(args, model, features, tag="dev", output_logits=False, inference_th=1.0):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    logits = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  'inference_th':inference_th
                  }

        with torch.no_grad():
            pred, logit = model(**inputs)
            #print(pred.size())
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
            logits.append(logit)
            #exit()

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    ans = to_official(preds, features)
    if tag == 'dev':
        f_name = args.dev_file
    elif tag == 'test':
        f_name = args.test_file
    if True:
        best_f1, _, best_f1_ign, _ , best_p , best_r, freq_F1, long_tail_F1, intra_F1, inter_F1, re_p_freq, re_r_freq, re_p_long_tail, re_r_long_tail  = official_evaluate_benchmark(ans, args.data_dir, args.train_file, f_name)
    output = {
        tag + "_P": best_p * 100,
        tag + "_R": best_r * 100,
        tag + "_F1": best_f1 * 100,
        tag + "_F1_ign": best_f1_ign * 100,
        tag + "_Freq_P": re_p_freq * 100,
        tag + "_Freq_R": re_r_freq * 100,
        tag + "_Freq_F1": freq_F1 * 100,
        tag + "_LT_P": re_p_long_tail * 100,  
        tag + "_LT_R": re_r_long_tail * 100,  
        tag + "_LT_F1": long_tail_F1 * 100,  
        #tag + "_Intra_F1": intra_F1 * 100,
        #tag + "_Inter_F1": inter_F1 * 100,

    }
    if output_logits==False:
        return best_f1, output
    elif output_logits==True:
        return best_f1, output, logits

def evaluate_by_class(args, model, features, data, tag="dev", output_logits=False, inference_th=1.0):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    predictions = []
    for step, batch in enumerate(dataloader):
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  'inference_th':inference_th,
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            #preds.append(pred)

        batch_features = features[step * args.test_batch_size : (step+1) * args.test_batch_size]
        label_idx = 0
        for feat_id, old_feature in enumerate(batch_features):    
            feature = deepcopy(old_feature)
            d_labels = old_feature['labels']
            pair_num = len(d_labels)
            #print(pred.size())
            prediction = pred[label_idx : label_idx + pair_num, :]
            prediction = prediction.cpu().numpy()
            prediction[np.isnan(prediction)] = 0
            #prediction[:, 0] = 0
            predictions.append(prediction)
            label_idx = label_idx + pair_num
    
    predictions = to_official_by_doc(predictions, features)
    scores_by_class = score_predictions_by_class(data, predictions)
    return  scores_by_class  
    
    



def generate_mixup_label(args, model, features, tag="distant", w_self= 0.7, w_distant = 0.3):
    new_features = []
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    logits = []
    for step, batch in enumerate(dataloader):
        
        model.eval()
        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred, logit = model(**inputs)
            #print(pred.size())
        batch_features = features[step * args.test_batch_size : (step+1) * args.test_batch_size]
        label_idx = 0
        for feat_id, old_feature in enumerate(batch_features):
            feature = deepcopy(old_feature)
            old_labels = old_feature['labels']
            pair_num = len(old_labels)
            #print(pair_num)
            
            tmp_labels = w_self * pred[label_idx : label_idx + pair_num, :].clone() + w_distant * old_labels.to(pred)
            feature['labels'] = tmp_labels.clamp(min=0.0, max=1.0) 
            #print(feature['labels'].size())
            assert len(old_labels) == feature['labels'].size()[0]
            label_idx = label_idx + pair_num
            new_features.append(feature)
    return new_features        



def generate_and_filter_label(args, model, features, tag="distant", w_p= 1.0, w_d = 1.0, d_drop_ratio = 0.0, p_drop_ratio = 0.0, d_rand_keep=1.0, p_rand_keep=1.0, inference_th=1.0):
    new_features = []
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    logits = []
    for step, batch in enumerate(dataloader):
        
        model.eval()
        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  'inference_th': inference_th,
                  }

        with torch.no_grad():
            pred, logit = model(**inputs)
            #print(pred.size())
        batch_features = features[step * args.test_batch_size : (step+1) * args.test_batch_size]
        label_idx = 0
        for feat_id, old_feature in enumerate(batch_features):
            feature = deepcopy(old_feature)
            d_labels = old_feature['labels']
            d_labels[:, 0] = 0 
            d_label_ind_0,  d_label_ind_1= torch.where(d_labels == 1)
            pair_num = len(d_labels)
            d_pos = len(d_label_ind_0)
            feature_logit = logit[label_idx : label_idx + pair_num, :].clone()
            feature_logit = feature_logit/feature_logit[:,:1] 
            d_label_logits = feature_logit[d_label_ind_0, d_label_ind_1]
            
            p_labels = pred[label_idx : label_idx + pair_num, :].clone()
            p_labels[:, 0] = 0
            p_label_ind_0,  p_label_ind_1= torch.where(p_labels == 1)
            p_pos = len(p_label_ind_0)
            
            p_label_logits = feature_logit[p_label_ind_0,  p_label_ind_1]
            
            #Low confidence Distant Labels
            d_logit_rank = torch.argsort(d_label_logits, descending=False)
            
            #Low confidence Pseudo Labels
            p_logit_rank = torch.argsort(p_label_logits, descending=False)
            
            d_drop_number = int(d_drop_ratio * d_pos)
            p_drop_number = int(p_drop_ratio * p_pos)
            
            d_drop_rand = int(  d_pos * (1 - d_rand_keep))
            p_drop_rand = int(  p_pos * (1 - p_rand_keep))
           
            
            d_id_rand = torch.randperm(len(d_logit_rank))[:d_drop_rand].to(d_logit_rank)
            p_id_rand = torch.randperm(len(p_logit_rank))[:p_drop_rand].to(p_logit_rank)
            
            
           
            d_id_rand = d_logit_rank[d_id_rand]
            p_id_rand = p_logit_rank[p_id_rand]
            
            d_drop_id = d_logit_rank[:d_drop_number]
            p_drop_id = p_logit_rank[:p_drop_number]
            
            d_drop_id = torch.cat([d_drop_id, d_id_rand])
            p_drop_id = torch.cat([p_drop_id, p_id_rand])
            #print(feature_logit[d_label_ind_0[d_drop_id], d_label_ind_1[d_drop_id]])
            #print(feature_logit[p_label_ind_0[p_drop_id], p_label_ind_1[p_drop_id]])
            #random dropping of labels
            d_labels[d_label_ind_0[d_drop_id], d_label_ind_1[d_drop_id]] = 0  
            p_labels[p_label_ind_0[p_drop_id], p_label_ind_1[p_drop_id]] = 0 
            
            #Mixup Pseudo labels and distant labels
            tmp_labels = w_p * p_labels  + w_d * d_labels.to(pred)
            feature['labels'] = tmp_labels.clamp(min=0.0, max=1.0) 
            #print(feature['labels'].size())
            assert len(d_labels) == feature['labels'].size()[0]
            label_idx = label_idx + pair_num
            new_features.append(feature)
          
            
    return new_features


#@profile()
def generate_class_balance_label(args, model, features, tag="distant", w_p= 1.0, w_d = 1.0, d_drop_ratio = 0.0, p_drop_ratio = 0.0, d_rand_keep=1.0, p_rand_keep=1.0, probs=1.0, inference_th=1.0):
    new_features = []
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    logits = []
    for step, batch in enumerate(dataloader):
        #if step==500:
        #    break
        model.eval()
        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  'inference_th': inference_th,
                  }

        with torch.no_grad():
            pred, logit = model(**inputs)
            pred = pred.detach().cpu().numpy()
            logit = logit.detach().cpu().numpy()
            #print(pred.size())
        batch_features = features[step * args.test_batch_size : (step+1) * args.test_batch_size]
        label_idx = 0
        for feat_id, old_feature in enumerate(batch_features):
            feature = deepcopy(old_feature)
            if type(old_feature['labels'])==list:
                d_labels = np.array(old_feature['labels'])
            else:
                d_labels = old_feature['labels']
            d_labels[:, 0] = 0 
            d_label_idx = np.argwhere(d_labels == 1)
            pair_num = len(d_labels)
            d_pos = len(d_label_idx)
            feature_logit = logit[label_idx : label_idx + pair_num, :]
            feature_logit = feature_logit/feature_logit[:,:1] 
            #print(feature_logit.shape)
            #print(d_label_idx.shape)
            d_label_logits = feature_logit[d_label_idx[:, 0], d_label_idx[:, 1]]
            
            p_labels = pred[label_idx : label_idx + pair_num, :]
            p_labels[:, 0] = 0
            p_label_idx= np.argwhere(p_labels == 1)
            p_pos = len(p_label_idx)
            
            p_label_logits = feature_logit[p_label_idx[:, 0], p_label_idx[:, 1]]
            

            
            #Pseudo label keep probability mask
            pseudo_mask = np.random.rand(p_labels.shape[0], p_labels.shape[1])

            pseudo_mask = np.less(pseudo_mask, probs)

            tmp_labels = w_p * p_labels * pseudo_mask  + w_d * d_labels
            feature['labels'] = tmp_labels.clip(min=0.0, max=1.0) 
            #print(feature['labels'].size())
            assert len(d_labels) == feature['labels'].shape[0]
            label_idx = label_idx + pair_num
            new_features.append(feature)
          
            
    return new_features

def find_hard_negatives(args, model, features, tag="distant", neg_rate=1.0, probs=1.0, inference_th=1.0):
    new_features = []
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    logits = []
    for step, batch in enumerate(dataloader):
        #if step==500:
        #    break
        model.eval()
        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  'inference_th': inference_th,
                  }

        with torch.no_grad():
            pred, logit = model(**inputs)
            pred = pred.detach().cpu()
            logit = logit.detach().cpu()
            num_class = logit.size()[1]
            th_mask = torch.cat( num_class * [logit[:,:1]], dim=1)
            logit_th = torch.cat([logit.unsqueeze(1), th_mask.unsqueeze(1)], dim=1) 
            
            log_probs = F.log_softmax(logit_th, dim=1)
            probs = torch.exp(F.log_softmax(logit_th, dim=1))
            log_prob_1 = log_probs[:, 0 ,:]
            prob_1 = probs[:, 0 ,:]
            entropy = - log_prob_1 * prob_1
            entropy = entropy.numpy()
            log_prob_1 = log_prob_1.numpy()
            prob_1 = prob_1.numpy()
            pred = pred.numpy()
            logit = logit.numpy()
            #print(pred.size())
        batch_features = features[step * args.test_batch_size : (step+1) * args.test_batch_size]
        label_idx = 0
        for feat_id, old_feature in enumerate(batch_features):
            feature = deepcopy(old_feature)
            if type(old_feature['labels'])==list:
                labels = np.array(old_feature['labels'])
            else:
                labels = old_feature['labels']
            labels[:, 0] = 0 
            neg_labels = labels.sum(axis=1, keepdims=False)
            pos_idx = np.argwhere(neg_labels!=0).squeeze()
            neg_idx = np.argwhere(neg_labels==0).squeeze()

            pair_num = len(labels)
            #feature_logit = logit[label_idx : label_idx + pair_num, :]
            #feature_logit = feature_logit/feature_logit[:,:1] 
            
            feature_entropy = entropy[label_idx : label_idx + pair_num, :]            
            neg_entropy = feature_entropy[neg_idx]
            mean_pos_score = neg_entropy[:, 1:].sum(axis=1)
            
            
            feature_logp = log_prob_1[label_idx : label_idx + pair_num, :]            
            neg_logp = feature_logp[neg_idx]
            max_pos_score = neg_logp[:, 1:].max(axis=1)
            
            
            ranked_neg_idx_mean = np.argsort(mean_pos_score)
            #sampled_neg_idx = neg_idx[ranked_neg_idx[-int(neg_rate * len(neg_idx)):]]
            #sampled_neg_idx_mean = neg_idx[ranked_neg_idx_mean[:int(neg_rate * len(neg_idx))]]
            sampled_neg_idx_mean = neg_idx[ranked_neg_idx_mean[int((0.05) * len(neg_idx)):int((neg_rate + 0.1) * len(neg_idx))]]
            
            ranked_neg_idx_max = np.argsort(max_pos_score)
            #sampled_neg_idx_max = neg_idx[ranked_neg_idx_max[:int(neg_rate * len(neg_idx))]]
            #sampled_neg_idx_max = neg_idx[ranked_neg_idx_max[-int(neg_rate * len(neg_idx)):]]
            sampled_neg_idx_max = neg_idx[ranked_neg_idx_max[int((0.05) * len(neg_idx)):int((neg_rate + 0.1) * len(neg_idx))]]

            sampled_neg_idx = np.intersect1d(sampled_neg_idx_max, sampled_neg_idx_mean)

            
            
            use_idx = np.hstack([pos_idx, sampled_neg_idx])
            #print(len(use_idx))
            label_idx = label_idx + pair_num
            #print(feature_logit.shape)
            if type(old_feature['labels']) == list:
                hts = np.array(old_feature['hts'])[use_idx]
                labels = np.array(old_feature['labels'])[use_idx]
                #print(use_idx.shape)
                #print(labels.shape)
        
            else:
                labels = old_feature['labels'][use_idx]
                #print(len(labels))
                hts = old_feature['hts'][use_idx]
                #print(hts.shape)
                #print(labels.shape)
            feature['labels'] = labels
            feature['hts'] = hts
            
            new_features.append(feature)
          
            
    return new_features


#@profile()
def generate_class_balance_label_pt(args, model, features, tag="distant", w_p= 1.0, w_d = 1.0, d_drop_ratio = 0.0, p_drop_ratio = 0.0, d_rand_keep=1.0, p_rand_keep=1.0, probs=1.0, inference_th=1.0):
    new_features = []
    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    logits = []
    for step, batch in enumerate(dataloader):
        #if step==500:
        model.eval()
        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  'inference_th': inference_th,
                  }

        with torch.no_grad():
            pred, logit = model(**inputs)
            #print(pred.size())
        batch_features = features[step * args.test_batch_size : (step+1) * args.test_batch_size]
        label_idx = 0
        for feat_id, old_feature in enumerate(batch_features):
            feature = deepcopy(old_feature)
            if type(old_feature['labels'])==list:
                d_labels = torch.tensor(old_feature['labels'])
            else:
                d_labels = old_feature['labels']
            d_labels[:, 0] = 0 
            d_label_ind_0,  d_label_ind_1= torch.where(d_labels == 1)
            pair_num = len(d_labels)
            d_pos = len(d_label_ind_0)
            feature_logit = logit[label_idx : label_idx + pair_num, :].detach().cpu().clone()
            feature_logit = feature_logit/feature_logit[:,:1] 
            d_label_logits = feature_logit[d_label_ind_0, d_label_ind_1]
            
            p_labels = pred[label_idx : label_idx + pair_num, :].detach().cpu().clone()
            p_labels[:, 0] = 0
            p_label_ind_0,  p_label_ind_1= torch.where(p_labels == 1)
            p_pos = len(p_label_ind_0)
            
            p_label_logits = feature_logit[p_label_ind_0,  p_label_ind_1]
            
            #Low confidence Distant Labels
            d_logit_rank = torch.argsort(d_label_logits, descending=False)
            
            #Low confidence Pseudo Labels
            p_logit_rank = torch.argsort(p_label_logits, descending=False)
            
            d_drop_number = int(d_drop_ratio * d_pos)
            p_drop_number = int(p_drop_ratio * p_pos)
            
            d_drop_rand = int(d_pos * (1 - d_rand_keep))
            p_drop_rand = int(p_pos * (1 - p_rand_keep))
           
            
            d_id_rand = torch.randperm(len(d_logit_rank))[:d_drop_rand].to(d_logit_rank)
            p_id_rand = torch.randperm(len(p_logit_rank))[:p_drop_rand].to(p_logit_rank)
            
            #Pseudo label keep probability mask
            pseudo_mask = torch.rand(p_labels.size())
            

            pseudo_mask = torch.le(pseudo_mask, probs).cpu()

            tmp_labels = w_p * p_labels * pseudo_mask  + w_d * d_labels
            feature['labels'] = tmp_labels.detach().clamp(min=0.0, max=1.0) 
            assert len(d_labels) == feature['labels'].size()[0]
            label_idx = label_idx + pair_num
            new_features.append(feature)
          
            
    return new_features


#@profile()
def get_random_mask(train_features, drop_prob, device):
    new_features = []
    device= torch.device('cpu')
    for feat_id, old_feature in enumerate(train_features):
        #if feat_id==100:
        feature = deepcopy(old_feature)
        if type(feature['labels']) == list:
            neg_labels = np.array(feature['labels']).sum(axis=1, keepdims=False)
            #print(neg_labels.shape)
        else:
            neg_labels = old_feature['labels'].sum(axis=1, keepdims=False)
        neg_index = np.argwhere(neg_labels==0).squeeze()
        #print(neg_index.shape)
        
        pos_index = np.argwhere(neg_labels!=0).squeeze()
        

        #print(neg_index[:10])
        sampled_negative_index = np.random.choice(neg_index, size=int((1-drop_prob) * len(neg_index)))
        #print(len(sampled_negative_index))


        use_idx = np.hstack([pos_index, sampled_negative_index])
        #print(len(use_idx))
        #assert len(use_idx) == (len(neg_labels) - len(sampled_negative_index))

            
        if type(old_feature['labels']) == list:
            hts = np.array(old_feature['hts'])[use_idx]
            labels = np.array(old_feature['labels'])[use_idx]
            #print(use_idx.shape)
            #print(labels.shape)
        
        else:
            labels = old_feature['labels'][use_idx]
            #print(len(labels))
            hts = old_feature['hts'][use_idx]
            #print(hts.shape)
            #print(labels.shape)
        feature['labels'] = labels
        feature['hts'] = hts
        new_features.append(feature)

    return new_features

def get_random_mask_pt(train_features, drop_prob, device):
    new_features = []
    device= torch.device('cpu')
    for feat_id, old_feature in enumerate(train_features):
        #if feat_id==100:
        feature = deepcopy(old_feature)
        if type(feature['labels']) == list:
            neg_labels = torch.tensor(feature['labels']).sum(dim=1)
        else:
            neg_labels = old_feature['labels'].clone().sum(dim=1)
        neg_index = torch.where(neg_labels==0)[0]
        pos_index = torch.where(neg_labels!=0)[0]
        #print(len(pos_index))
        #print(len(neg_index))
        perm = torch.randperm(neg_index.size(0))
        sampled_negative_index = neg_index[perm[:int(drop_prob * len(neg_index))]]
        #print(len(perm))
        #sampled_negative_index = neg_index
        neg_mask = torch.ones(len(feature['labels']))
        neg_mask[sampled_negative_index] = 0
        #feature['negative_mask'] = neg_mask        
        use_idx = torch.where(neg_mask==1)[0]

            
        if type(old_feature['labels']) == list:
            hts = torch.LongTensor(old_feature['hts'])[use_idx].to(device)
            labels = torch.LongTensor(old_feature['labels'])[use_idx].to(device)
            #print(len(labels))
        
        else:
            labels = old_feature['labels'][use_idx].clone().to(device)
            #print(len(labels))
            hts = old_feature['hts'].clone()[use_idx].to(device)
        feature['labels'] = labels
        feature['hts'] = hts
        new_features.append(feature)
        #exit()

    return new_features

def report(args, model, features, by_doc=False , inference_th=1.0):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds = []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  'inference_th':inference_th,
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)

    
    if by_doc==False:
        preds = np.concatenate(preds, axis=0).astype(np.float32)
        preds = to_official(preds, features)
    if by_doc==True:
        print(len(preds))
        preds = to_official_by_doc(preds, features)
        print(len(preds))
    return preds

def calculate_probability(scores_by_class, gamma_p=1.0, gamma_r=1.0):
    prob = np.ones(99)
    for item in scores_by_class:
        prob[rel2id[item['type']]] = ((float(item['precision'])/100) ** gamma_p) * ((1 - float(item['recall'])/100) ** gamma_r)
    print(prob)
    return prob


def calculate_probability_crest(crest_probs):
    prob = np.ones(99)
    for item in crest_probs:
        prob[item[0]] = item[1]
    return prob


def calculate_dynamic_thresholds(scores_by_class, beta_p=1.0, beta_r=1.0):
    threshold = torch.ones(99)
    base_threshods = 1.0
    #Betas - smoothing factors 
    for item in scores_by_class:
        threshold[rel2id[item['type']]] = ((float(item['recall'])/100) + beta_p ) / ((float(item['precision'])/100) + beta_r) 
    print(threshold)
    threshold = torch.clamp(threshold, min=0.85, max=1.05)
    print(threshold)
    return threshold


def calculate_probability_pt(scores_by_class, gamma_p=1.0, gamma_r=1.0):
    prob = torch.ones(99)
    for item in scores_by_class:
        prob[rel2id[item['type']]] = ((float(item['precision'])/100) ** gamma_p) * ((1 - float(item['recall'])/100) ** gamma_r)
    print(prob)
    return prob
    


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/docred", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str)
    parser.add_argument("--cache_dir", default="cache", type=str)

    parser.add_argument("--train_file", default="train_annotated.json", type=str)
    parser.add_argument("--distant_file", default="train_distant_10k.json", type=str)
    parser.add_argument("--dev_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)
    parser.add_argument("--load_train_features", default="", type=str)
    parser.add_argument("--load_pretrained", default="", type=str)
    parser.add_argument("--infer_path", default="", type=str)
    parser.add_argument("--output_name", default="result.json", type=str)


    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--n_rounds", default=5, type=int,
                        help="Number of Rounds for self-training.")
    parser.add_argument("--n_splits", default=5, type=int,
                        help="Number of Splits for self-training")

    parser.add_argument("--start_round", default=0, type=int,
                        help="Number of starting round for self training.")
    
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=4, type=int,
                        help="Max number of labels in prediction.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--classifier_lr", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--inference_th", default=1.0, type=float,
                        help="Scaling factor for threshold class during pseudo-labeling.")
    parser.add_argument("--crest", default=False, type=bool,
                        help="Flag for CREST training")
    parser.add_argument("--drop_prob", default=0.0, type=float,
                        help="Negative Drop Rate")
    parser.add_argument("--d_drop_ratio", default=0.0, type=float,
                        help="Ratio to discard document label by confidence")
    parser.add_argument("--p_drop_ratio", default=0.0, type=float,
                        help="Ratio to discard pseudo label by confidence")
    parser.add_argument("--d_rand_keep", default=1.0, type=float,
                        help="Ratio to randomly keep document label (noise)")
    parser.add_argument("--p_rand_keep", default=1.0, type=float,
                        help="Ratio to randomly keep document label (noise)")
    parser.add_argument("--w_self", default=1.0, type=float,
                        help="Weight for pseudo labels")
    parser.add_argument("--w_distant", default=1.0, type=float,
                        help="Weight for distant labels")
    parser.add_argument("--gamma_p", default=1.0, type=float,
                        help="gamma for precision")
    parser.add_argument("--gamma_r", default=1.0, type=float,
                        help="gamma for recall")
    parser.add_argument("--beta_p", default=1.0, type=float,
                        help="beta for precision")
    parser.add_argument("--beta_r", default=1.0, type=float,
                        help="beta for recall")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=97,
                        help="Number of relation types in dataset.")
    parser.add_argument("--chunk_size", type=int, default=10000,
                        help="Number of instance per iteration")
    parser.add_argument("--re_init", type=bool, default=True,
                        help="Re-initialization for each round")
    parser.add_argument("--iteration_epoch", type=int, default=20,
                        help="training epoch for each data integration")

    args = parser.parse_args()
    #wandb.init(project="DocRED", mode='disabled')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
        cache_dir=args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir,
    )

    read = read_docred

    train_file = os.path.join(args.data_dir, args.train_file)
    distant_file = os.path.join(args.data_dir, args.distant_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    dev_data = json.load(open(dev_file))
    train_features = read(train_file, tokenizer, max_seq_length=args.max_seq_length)
    distant_features = read(distant_file, tokenizer, max_seq_length=args.max_seq_length)
    dev_features = read(dev_file, tokenizer, max_seq_length=args.max_seq_length)
    test_features = read(test_file, tokenizer, max_seq_length=args.max_seq_length)
    #train_features = get_random_mask(train_features, 0.0, args.device)
    train_features = get_random_mask(train_features, args.drop_prob , args.device)
    distant_features = get_random_mask(distant_features, 0.0, args.device )
    
    if args.load_train_features != '':
        del train_features
        train_features = torch.load(args.load_train_features)

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type
    

    set_seed(args)
    model = DocREModel(config, model_name=args.model_name_or_path, model=None, cache_dir=args.cache_dir, num_labels=args.num_labels)
    model.to(0)

    if args.load_path == "" and args.load_pretrained == "" and args.infer_path == "":  # Training
        Multi_Round_KF_self_training(args, model, config, train_features, distant_features, dev_features, test_features, n_splits=args.n_splits, n_rounds=args.n_rounds)
    elif args.load_pretrained != "":  # Training
        model = amp.initialize(model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(args.load_pretrained), strict=False)
        print('Loaded Model')
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
        print(dev_output)
        train(args, model, train_features, dev_features, test_features)
    elif args.infer_path != "":  # Training

        train(args, model, config, train_features, dev_features, test_features)
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev", output_logits=False, inference_th=args.inference_th)
        print(dev_output)
        scores_by_class = evaluate_by_class(args, model, dev_features, dev_data, tag="dev", output_logits=False, inference_th=args.inference_th)
        probs = calculate_probability(scores_by_class, args.gamma_p, args.gamma_r)
        threshold = calculate_dynamic_thresholds(scores_by_class, args.beta_p, args.beta_r)
        scores_by_class = evaluate_by_class(args, model, dev_features, dev_data, tag="dev", output_logits=False, inference_th=threshold)

        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev", output_logits=False, inference_th=threshold)
        print(dev_output)
        model = DocREModel(config, args.model_name_or_path, args.cache_dir, num_labels=args.num_labels)
        model.to(0)
        print('Training a newly initialized model')
        #distant_features = get_random_mask(distant_features, args.drop_prob, args.device )
        train_features = get_random_mask(train_features, 0.93, args.device)
        train(args, model, config, train_features, dev_features, test_features)
        model.load_state_dict(torch.load(args.save_path), strict=False)
        
        scores_by_class = evaluate_by_class(args, model, dev_features, dev_data, tag="dev", output_logits=False, inference_th=args.inference_th)
        probs = calculate_probability(scores_by_class, args.gamma_p, args.gamma_r)
        threshold = calculate_dynamic_thresholds(scores_by_class, args.beta_p, args.beta_r)
        scores_by_class = evaluate_by_class(args, model, dev_features, dev_data, tag="dev", output_logits=False, inference_th=threshold)

    elif args.load_path != "" and args.load_pretrained == "":  # Testing
        model = amp.initialize(model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(args.load_path), strict=False)
        #pos_l, neg_l = get_losses(args, model, train_features, tag="train")
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev", inference_th=args.inference_th)
        test_score, test_output = evaluate(args, model, test_features, tag="test", inference_th=args.inference_th)
        print(dev_output)
        print(test_output)
        pred = report(args, model, dev_features, by_doc=False, inference_th=args.inference_th)
        with open(args.output_name, "w") as fh:
            json.dump(pred, fh)


if __name__ == "__main__":
    main()
