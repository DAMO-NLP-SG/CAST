import argparse
import os
import json
import numpy as np
import torch
from apex import amp
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model import DocREModel
from utils import set_seed, collate_fn
from evaluation import to_official_by_doc_bio, score_predictions_by_class_bio
from prepro import read_cdr, read_gda, read_chemdisgene
from copy import deepcopy
ctd_rel2id = {
	"chem_disease:marker/mechanism": 1,
	"chem_disease:therapeutic": 2,
	"chem_gene:increases^expression": 3,
	"chem_gene:decreases^expression": 4,
	"gene_disease:marker/mechanism": 5,
	"chem_gene:increases^activity": 6,
	"chem_gene:decreases^activity": 7,
	"chem_gene:increases^metabolic_processing": 8,
	"chem_gene:affects^binding": 9,
	"chem_gene:increases^transport": 10,
	"chem_gene:decreases^metabolic_processing": 11,
	"chem_gene:affects^localization": 12,
	"chem_gene:affects^expression": 13,
	"gene_disease:therapeutic": 14
}

id2rel_ctd = {value: key for key, value in ctd_rel2id.items()}

crest_probs = [(3, 0.30209255547153796), (1, 0.4175383280976381), (4, 0.4200262175606844), (5, 0.4558561175762247), (2, 0.5013507798105281), (8, 0.5155867627910126), (6, 0.5155867627910126), (7, 0.5522636550541499), (11, 0.6080184545917514), (10, 0.6766862458035041), (9, 0.714013960230954), (13, 0.8229580879560806), (12, 0.8614476467316166), (14, 1.0)]

def train(args, model, train_features, dev_features, test_features):
    def finetune(features, optimizer, num_epoch, num_steps):
        best_score = -1
        print(len(features))
        train_dataloader = DataLoader(features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=False)
        print(len(train_dataloader))
        
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
                outputs = model(**inputs)
                loss = outputs[0] / args.gradient_accumulation_steps
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if step % args.gradient_accumulation_steps == 0:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    num_steps += 1
                if (step + 1) == len(train_dataloader) - 1 or (args.evaluation_steps > 0 and num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                    dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
                    test_score, test_output = evaluate(args, model, test_features, tag="test")
                    print(dev_output)
                    print(test_output)
                    if dev_score > best_score:
                        best_score = dev_score
                        if args.save_path != "":
                            torch.save(model.state_dict(), args.save_path)

        return num_steps

    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": 1e-4},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
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
                dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
                #wandb.log(dev_output, step=num_steps)
                print(dev_output)
                if dev_score > best_score:
                    best_score = dev_score
                    best_model = deepcopy(model)
                    test_score, test_output = evaluate(args, model, test_features, tag="dev")
                    print(test_output)
                    if args.save_path != "":
                        torch.save(model.state_dict(), args.save_path)
    return best_model
    
    
def train_self_training(args, model, config, train_features, dev_features, test_features):


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
    num_splits = int(math.ceil(len(train_features)/chunk_size))
    temp_epoch = args.iteration_epoch
    for i in range(num_splits):
        print('Pseudo-label Round: {}'.format(i + 1))
        temp_distant = generate_mixup_label(args, model, train_features[i * chunk_size: (i + 1) * chunk_size], 
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

def train_KF_self_training(args, config, train_features, dev_features, dev_rels, test_features, n_splits):

    global crest_probs 
    splits = n_splits 
    trains, devs = get_train_dev_split(train_features, args.n_splits)
    merged_train = []
    #dev_file = os.path.join(args.data_dir, args.dev_file)
    #dev_data = json.load(open(dev_file))
    for i in range(splits):
        model, optimizer = initialize_model_and_optimizer(args, config)
        num_steps = 0
        set_seed(args)
        model.zero_grad()
        #trains[i] = get_random_mask(trains[i], args.drop_prob, args.device)
        model = finetune_func(args, model, trains[i], dev_features, test_features, optimizer, args.num_train_epochs, num_steps)
        scores_by_class = evaluate_by_class(args, model, dev_features, dev_rels, tag="dev", output_logits=False, inference_th = args.inference_th)
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
    if args.dlc==True:
        dlc_dir='/root/data/project/docred_production/ATLOP/'
    else:
        dlc_dir='./'
    encoder = AutoModel.from_pretrained(dlc_dir + 'BiomedNLP-PubMedBERT-base-uncased-abstract/')
    model = DocREModel(config, model_name='', model=encoder, cache_dir=args.cache_dir, num_labels=args.num_labels).to(args.device)
    new_layer = ["extractor", "bilinear"]
    optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in new_layer)], },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in new_layer)], "lr": args.classifier_lr},
            ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)
    return model, optimizer


def Multi_Round_KF_self_training(args, model, config, train_features,  dev_features, dev_rels, test_features, n_splits, n_rounds):
    for i in range(args.start_round, args.start_round + n_rounds):
        train_features = train_KF_self_training(args, config, train_features,  dev_features, dev_rels, test_features, n_splits)
        torch.save(train_features, '{}-round-{}-drop-{}-final-{}-rounds-{}-splits-w_self-{}-w_distant-{}-d_drop_ratio-{}-p_drop_ratio-{}-gamma_p-{}-gamma_r-{}.features'.format(args.save_path[:-3],  i, args.drop_prob, args.n_rounds, args.n_splits, args.w_self, args.w_distant, args.d_drop_ratio, args.p_drop_ratio, args.gamma_p, args.gamma_r ))
        model, optimizer = initialize_model_and_optimizer(args, config)
        num_steps = 0
        set_seed(args)
        model.zero_grad()
        #sampled_train_features = get_random_mask(train_features, args.drop_prob, args.device)
        print('Final training for Round: {} Neg Sample rate: {}'.format(i+1, 1 - args.drop_prob))
        model = finetune_func(args, model, train_features, dev_features, test_features, optimizer, args.num_train_epochs, num_steps)
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
        torch.save(model.state_dict(),'{}-round-{}-drop-{}-final-{}-rounds-{}-splits-w_self-{}-w_distant-{}-d_drop_ratio-{}-p_drop_ratio-{}-gamma_p-{}-gamma_r-{}.pt'.format(args.save_path[:-3],  i, args.drop_prob, args.n_rounds, args.n_splits, args.w_self, args.w_distant, args.d_drop_ratio, args.p_drop_ratio, args.gamma_p, args.gamma_r ))
        del model, optimizer
        print('KFold Self Training Round {} Result:'.format(i+1))
        print(dev_output)
    return None    
    
    
    
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
    
    predictions = to_official_by_doc_bio(predictions, features)
    scores_by_class = score_predictions_by_class_bio(data, predictions)
    return  scores_by_class      

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
            #print(pseudo_mask[1])
            #exit()
            #Mixup Pseudo labels and distant labels
            #print(type(d_labels))
            tmp_labels = w_p * p_labels * pseudo_mask  + w_d * d_labels
            feature['labels'] = tmp_labels.clip(min=0.0, max=1.0) 
            #print(feature['labels'].size())
            assert len(d_labels) == feature['labels'].shape[0]
            label_idx = label_idx + pair_num
            new_features.append(feature)
          
            
    return new_features

def evaluate(args, model, features, tag="dev"):

    dataloader = DataLoader(features, batch_size=args.test_batch_size, shuffle=False, collate_fn=collate_fn, drop_last=False)
    preds, golds = [], []
    for batch in dataloader:
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'entity_pos': batch[3],
                  'hts': batch[4],
                  }

        with torch.no_grad():
            pred, *_ = model(**inputs)
            pred = pred.cpu().numpy()
            pred[np.isnan(pred)] = 0
            preds.append(pred)
            golds.append(np.concatenate([np.array(label, np.float32) for label in batch[2]], axis=0))

    preds = np.concatenate(preds, axis=0).astype(np.float32)
    golds = np.concatenate(golds, axis=0).astype(np.float32)

    tp = ((preds[:, 1] == 1) & (golds[:, 1] == 1)).astype(np.float32).sum()
    tn = ((golds[:, 1] == 1) & (preds[:, 1] != 1)).astype(np.float32).sum()
    fp = ((preds[:, 1] == 1) & (golds[:, 1] != 1)).astype(np.float32).sum()
    precision = tp / (tp + fp + 1e-5)
    recall = tp / (tp + tn + 1e-5)
    f1 = 2 * precision * recall / (precision + recall + 1e-5)
    output = {
        "{}_p".format(tag): precision * 100,
        "{}_r".format(tag): recall * 100,
        "{}_f1".format(tag): f1 * 100,
    }
    return f1, output

def calculate_probability(scores_by_class, gamma_p=1.0, gamma_r=1.0):
    prob = np.ones(15)
    for item in scores_by_class:
        prob[ctd_rel2id[item['type']]] = ((float(item['precision'])/100) ** gamma_p) * ((1 - float(item['recall'])/100) ** gamma_r)
    print(prob)
    return prob


def calculate_probability_crest(crest_probs):
    prob = np.ones(15)
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
        #exit()

    return new_features


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./dataset/cdr", type=str)
    parser.add_argument("--transformer_type", default="bert", type=str)
    parser.add_argument("--cache_dir", default="cache", type=str)
    parser.add_argument("--model_name_or_path", default="allenai/scibert_scivocab_cased", type=str)

    parser.add_argument("--train_file", default="train_filter.data", type=str)
    parser.add_argument("--dev_file", default="dev_filter.data", type=str)
    parser.add_argument("--test_file", default="test_filter.data", type=str)
    parser.add_argument("--dlc", default=False, type=bool)
    parser.add_argument("--save_path", default="", type=str)
    parser.add_argument("--load_path", default="", type=str)

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--drop_prob", default=0.0, type=float,
                        help="Negative Drop Rate")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--num_labels", default=1, type=int,
                        help="Max number of labels in the prediction.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--classifier_lr", default=3e-5, type=float,
                        help="The initial learning rate for Adam for classifier.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.06, type=float,
                        help="Warm up ratio for Adam.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--evaluation_steps", default=-1, type=int,
                        help="Number of training steps between evaluations.")
    parser.add_argument("--seed", type=int, default=66,
                        help="random seed for initialization.")
    parser.add_argument("--num_class", type=int, default=2,
                        help="Number of relation types in dataset.")
    parser.add_argument("--inference_th", default=1.0, type=float,
                        help="Scaling factor for threshold class during pseudo-labeling.")
    parser.add_argument("--crest", default=False, type=bool,
                        help="Flag for CREST training")
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
    parser.add_argument("--n_rounds", default=5, type=int,
                        help="Number of Rounds for self-training.")
    parser.add_argument("--n_splits", default=2, type=int,
                        help="Number of Splits for self-training")

    parser.add_argument("--start_round", default=0, type=int,
                        help="Number of starting round for self training.")
    args = parser.parse_args()
    if args.dlc==True:
        dlc_dir='/root/data/project/docred_production/ATLOP/'
    else:
        dlc_dir='./'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(dlc_dir + 'BiomedNLP-PubMedBERT-base-uncased-abstract/')
    tokenizer = AutoTokenizer.from_pretrained(dlc_dir + 'BiomedNLP-PubMedBERT-base-uncased-abstract/')

    read = read_chemdisgene

    train_file = os.path.join(args.data_dir, args.train_file)
    dev_file = os.path.join(args.data_dir, args.dev_file)
    test_file = os.path.join(args.data_dir, args.test_file)
    dev_data = json.load(open(dev_file))
    test_data = json.load(open(test_file))
    
    train_features, train_rels = read(args, train_file, tokenizer, max_seq_length=args.max_seq_length)
    dev_features, dev_rels = read(args, dev_file, tokenizer, max_seq_length=args.max_seq_length)
    test_features, test_rels = read(args, test_file, tokenizer, max_seq_length=args.max_seq_length)
    train_features = get_random_mask(train_features, args.drop_prob, 0)
    #exit()
    encoder = AutoModel.from_pretrained(dlc_dir + 'BiomedNLP-PubMedBERT-base-uncased-abstract/')

    config.cls_token_id = tokenizer.cls_token_id
    config.sep_token_id = tokenizer.sep_token_id
    config.transformer_type = args.transformer_type

    set_seed(args)
    model = DocREModel(config, model_name='', model=encoder, cache_dir=args.cache_dir, num_labels=args.num_labels)
    model.to(0)

    if args.load_path == "":
        #train(args, model, train_features, dev_features, test_features)
        Multi_Round_KF_self_training(args, model, config, train_features, dev_features, dev_rels, test_features, n_splits=args.n_splits, n_rounds=args.n_rounds)
    else:
        model = amp.initialize(model, opt_level="O1", verbosity=0)
        model.load_state_dict(torch.load(args.load_path), strict=False)
        dev_score, dev_output = evaluate(args, model, dev_features, tag="dev")
        test_score, test_output = evaluate(args, model, test_features, tag="test")
        #scores = evaluate_by_class(args, model, test_features, test_rels, tag="test")
        print(dev_output)
        print(test_output)
        #print(scores)


if __name__ == "__main__":
    main()
