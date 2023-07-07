import os
import os.path
import json
import numpy as np
from itertools import accumulate
import sys
from typing import List, Tuple
from sklearn.metrics import precision_recall_fscore_support as prfs

METRIC_LABELS = ['prec_micro', 'rec_micro', 'f1_micro', 'prec_macro', 'rec_macro', 'f1_macro']

#rel2id = json.load(open('meta/rel2id.json', 'r'))
rel2id = {'P1376': 79, 'P607': 27, 'P136': 73, 'P137': 63, 'P131': 2, 'P527': 11, 'P1412': 38, 'P206': 33, 'P205': 77, 'P449': 52, 'P127': 34, 'P123': 49, 'P86': 66, 'P840': 85, 'P355': 72, 'P737': 93, 'P740': 84, 'P190': 94, 'P576': 71, 'P749': 68, 'P112': 65, 'P118': 40, 'P17': 1, 'P19': 14, 'P3373': 19, 'P6': 42, 'P276': 44, 'P1001': 24, 'P580': 62, 'P582': 83, 'P585': 64, 'P463': 18, 'P676': 87, 'P674': 46, 'P264': 10, 'P108': 43, 'P102': 17, 'P25': 81, 'P27': 3, 'P26': 26, 'P20': 37, 'P22': 30, 'Na': 0, 'P807': 95, 'P800': 51, 'P279': 78, 'P1336': 88, 'P577': 5, 'P570': 8, 'P571': 15, 'P178': 36, 'P179': 55, 'P272': 75, 'P170': 35, 'P171': 80, 'P172': 76, 'P175': 6, 'P176': 67, 'P39': 91, 'P30': 21, 'P31': 60, 'P36': 70, 'P37': 58, 'P35': 54, 'P400': 31, 'P403': 61, 'P361': 12, 'P364': 74, 'P569': 7, 'P710': 41, 'P1344': 32, 'P488': 82, 'P241': 59, 'P162': 57, 'P161': 9, 'P166': 47, 'P40': 20, 'P1441': 23, 'P156': 45, 'P155': 39, 'P150': 4, 'P551': 90, 'P706': 56, 'P159': 29, 'P495': 13, 'P58': 53, 'P194': 48, 'P54': 16, 'P57': 28, 'P50': 22, 'P1366': 86, 'P1365': 92, 'P937': 69, 'P140': 50, 'P69': 25, 'P1198': 96, 'P1056': 89}
id2rel = {value: key for key, value in rel2id.items()}

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

def score(gt: List[List[Tuple]], pred: List[List[Tuple]], type_idx=None, print_results: bool = False):
    assert len(gt) == len(pred)

    gt_flat = []
    pred_flat = []
    #labels= set()

    for (sample_gt, sample_pred) in zip(gt, pred):
        union = []
        for s in sample_gt:
            if s not in union:
                union.append(s)

        for s in sample_pred:
            if s not in union:
                union.append(s)

        for s in union:
            if s in sample_gt:
                if type_idx is not None:
                    t = s[type_idx]
                    gt_flat.append(rel2id[t])
                    
                    #labels.add(t)
                else:
                    gt_flat.append(0)
            else:
                gt_flat.append(-1)

            if s in sample_pred:
                if type_idx is not None:
                    #print(s)
                    #break
                    t = s[type_idx]
                    pred_flat.append(rel2id[t])
                    #labels.add(t)
                else:
                    pred_flat.append(0)
            else:
                pred_flat.append(-1)

    if type_idx is not None:
        if True:
            #print(labels)
            labels, labels_str = zip(*[( rel2id[i], i) for i in rel2id])
        else:
            # corner case: no prediction and ground truth samples
            labels, labels_str = [0], ['None']
    else:
        labels, labels_str = [0], ['Binary']

    metrics, per_type_result = _compute_metrics(gt_flat, pred_flat, labels, labels_str, print_results)
    return metrics, per_type_result




def score_bio(gt: List[List[Tuple]], pred: List[List[Tuple]], type_idx=None, print_results: bool = False):
    assert len(gt) == len(pred)

    gt_flat = []
    pred_flat = []
    #labels= set()

    for (sample_gt, sample_pred) in zip(gt, pred):
        union = []
        for s in sample_gt:
            if s not in union:
                union.append(s)

        for s in sample_pred:
            if s not in union:
                union.append(s)

        for s in union:
            if s in sample_gt:
                if type_idx is not None:
                    t = s[type_idx]
                    gt_flat.append(ctd_rel2id[t])
                    
                    #labels.add(t)
                else:
                    gt_flat.append(0)
            else:
                gt_flat.append(-1)

            if s in sample_pred:
                if type_idx is not None:
                    #print(s)
                    #break
                    t = s[type_idx]
                    pred_flat.append(ctd_rel2id[t])
                    #labels.add(t)
                else:
                    pred_flat.append(0)
            else:
                pred_flat.append(-1)

    if type_idx is not None:
        if True:
            #print(labels)
            labels, labels_str = zip(*[( ctd_rel2id[i], i) for i in ctd_rel2id])
        else:
            # corner case: no prediction and ground truth samples
            labels, labels_str = [0], ['None']
    else:
        labels, labels_str = [0], ['Binary']

    metrics, per_type_result = _compute_metrics(gt_flat, pred_flat, labels, labels_str, print_results)
    return metrics, per_type_result

def score_single(gt: List[Tuple], pred: List[Tuple], type_idx=None, print_results: bool = False):
    return score([gt], [pred], type_idx=type_idx, print_results=print_results)


def _compute_metrics(gt_all, pred_all, labels, labels_str, print_results: bool = False):
    per_type = prfs(gt_all, pred_all, labels=labels, average=None, zero_division=0)
    micro = prfs(gt_all, pred_all, labels=labels, average='micro', zero_division=0)[:-1]
    macro = prfs(gt_all, pred_all, labels=labels, average='macro', zero_division=0)[:-1]
    total_support = sum(per_type[-1])

    if print_results:
        per_type_result = _print_results(per_type, list(micro) + [total_support], list(macro) + [total_support], labels_str)

    metrics = [m * 100 for m in micro + macro]
    return dict(zip(METRIC_LABELS, metrics)), per_type_result


def _print_results(per_type: List, micro: List, macro: List, types: List):
    columns = ('type', 'precision', 'recall', 'f1-score', 'support')
    per_type_result = []
    row_fmt = "%20s" + (" %12s" * (len(columns) - 1))
    results = [row_fmt % columns, '\n']

    metrics_per_type = []
    for i, t in enumerate(types):
        metrics = []
        for j in range(len(per_type)):
            metrics.append(per_type[j][i])
        metrics_per_type.append(metrics)

    for m, t in zip(metrics_per_type, types):
        #print(_get_row(m, t))
        #print(type(_get_row(m, t)))
        tmp = _get_row(m, t)
        per_type_result.append({'type':tmp[0], 'precision':tmp[1], 'recall':tmp[2], 'f1-score':tmp[3], 'support':tmp[4]})
        results.append(row_fmt % _get_row(m, t))
        results.append('\n')

    results.append('\n')

    # micro
    results.append(row_fmt % _get_row(micro, 'micro'))
    results.append('\n')

    # macro
    results.append(row_fmt % _get_row(macro, 'macro'))

    results_str = ''.join(results)
    print(results_str)
    return per_type_result

def _get_row(data, label):
    row = [label]
    for i in range(len(data) - 1):
        row.append("%.2f" % (data[i] * 100))
    row.append(data[3])
    return tuple(row)

def map_ground_truth(data):
    subset_gt = []
    for example in data:
        exp_gt = []
        for label in example['labels']:
            exp_gt.append((label['h'], label['t'], label['r']))
        subset_gt.append(exp_gt)
    return subset_gt




def findSmallestDifference(A, B, m, n):
 
    # Sort both arrays
    # using sort function
    A.sort()
    B.sort()
 
    a = 0
    b = 0
 
    # Initialize result as max value
    result = sys.maxsize
 
    # Scan Both Arrays upto
    # sizeof of the Arrays
    while (a < m and b < n):
     
        if (abs(A[a] - B[b]) < result):
            result = abs(A[a] - B[b])
 
        # Move Smaller Value
        if (A[a] < B[b]):
            a += 1
 
        else:
            b += 1
    # return final sma result
    return result




def to_official(preds, features):
    h_idx, t_idx, title = [], [], []

    for f in features:
        hts = f["hts"]
        h_idx += [ht[0] for ht in hts]
        t_idx += [ht[1] for ht in hts]
        title += [f["title"] for ht in hts]

    res = []
    for i in range(preds.shape[0]):
        pred = preds[i]
        pred = np.nonzero(pred)[0].tolist()
        for p in pred:
            if p != 0 and p < 97:
                res.append(
                    {
                        'title': title[i],
                        'h_idx': h_idx[i],
                        't_idx': t_idx[i],
                        'r': id2rel[p],
                    }
                )
    return res

def to_official_by_doc(preds, features):
    h_idx, t_idx, title = [], [], []
    res = []
    for pred, f in zip(preds, features):
        hts = f["hts"]
        h_idx = [ht[0] for ht in hts]
        t_idx = [ht[1] for ht in hts]
        title = [f["title"] for ht in hts]
        local_res = []
        

        for i in range(pred.shape[0]):
            pred_i = np.nonzero(pred[i])[0].tolist()
            for p in pred_i:
                if p != 0 and p < 97:
                    local_res.append(
                        {
                            'title': title[i],
                            'h_idx': h_idx[i],
                            't_idx': t_idx[i],
                            'r': id2rel[p],
                        }
                    )
        res.append(local_res)
    return res

def to_official_by_doc_bio(preds, features):
    h_idx, t_idx, title = [], [], []
    res = []
    for pred, f in zip(preds, features):
        hts = f["hts"]
        h_idx = [ht[0] for ht in hts]
        t_idx = [ht[1] for ht in hts]
        title = [f["title"] for ht in hts]
        local_res = []
        

        for i in range(pred.shape[0]):
            pred_i = np.nonzero(pred[i])[0].tolist()
            for p in pred_i:
                if p != 0 and p < 97:
                    local_res.append(
                        {
                            'title': title[i],
                            'h_idx': h_idx[i],
                            't_idx': t_idx[i],
                            'r': id2rel_ctd[p],
                        }
                    )
        res.append(local_res)
    return res

def gen_train_facts(data_file_name, truth_dir):
    fact_file_name = data_file_name[data_file_name.find("train_"):]
    fact_file_name = os.path.join(truth_dir, fact_file_name.replace(".json", ".fact"))

    if os.path.exists(fact_file_name):
        fact_in_train = set([])
        triples = json.load(open(fact_file_name))
        for x in triples:
            fact_in_train.add(tuple(x))
        return fact_in_train

    fact_in_train = set([])
    ori_data = json.load(open(data_file_name))
    for data in ori_data:
        vertexSet = data['vertexSet']
        for label in data['labels']:
            rel = label['r']
            for n1 in vertexSet[label['h']]:
                for n2 in vertexSet[label['t']]:
                    fact_in_train.add((n1['name'], n2['name'], rel))

    json.dump(list(fact_in_train), open(fact_file_name, "w"))

    return fact_in_train


def official_evaluate(tmp, path, train_file, dev_file):
    '''
        Adapted from the official evaluation code
    '''
    truth_dir = os.path.join(path, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    fact_in_train_annotated = gen_train_facts(os.path.join(path, train_file), truth_dir)
    fact_in_train_distant = gen_train_facts(os.path.join(path, "train_distant.json"), truth_dir)

    truth = json.load(open(os.path.join(path, dev_file)))

    std = {}
    tot_evidences = 1
    titleset = set([])

    title2vectexSet = {}

    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            tot_evidences += len(label['evidence'])

    tot_relations = len(std)
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    if len(tmp)>0:
        submission_answer = [tmp[0]]
    else:
        return 0, 0, 0, 0 , 0, 0
    for i in range(1, len(tmp)):
        x = tmp[i]
        y = tmp[i - 1]
        if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
            submission_answer.append(tmp[i])

    correct_re = 0
    correct_evidence = 0
    pred_evi = 0

    correct_in_train_annotated = 0
    correct_in_train_distant = 0
    titleset2 = set([])
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if 'evidence' in x:
            evi = set(x['evidence'])
        else:
            evi = set([])
        pred_evi += len(evi)

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            stdevi = std[(title, r, h_idx, t_idx)]
            correct_evidence += len(stdevi & evi)
            in_train_annotated = in_train_distant = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                        in_train_annotated = True
                    if (n1['name'], n2['name'], r) in fact_in_train_distant:
                        in_train_distant = True

            if in_train_annotated:
                correct_in_train_annotated += 1
            if in_train_distant:
                correct_in_train_distant += 1

    re_p = 1.0 * correct_re / len(submission_answer)
    re_r = 1.0 * correct_re / tot_relations
    if re_p + re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

    evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
    evi_r = 1.0 * correct_evidence / tot_evidences
    if evi_p + evi_r == 0:
        evi_f1 = 0
    else:
        evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

    re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (len(submission_answer) - correct_in_train_annotated + 1e-5)
    re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / (len(submission_answer) - correct_in_train_distant + 1e-5)

    if re_p_ignore_train_annotated + re_r == 0:
        re_f1_ignore_train_annotated = 0
    else:
        re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

    if re_p_ignore_train + re_r == 0:
        re_f1_ignore_train = 0
    else:
        re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

    return re_f1, evi_f1, re_f1_ignore_train_annotated, re_f1_ignore_train , re_p, re_r



def official_evaluate_benchmark(tmp, path, train_file, dev_file):
    '''
        Adapted from the official evaluation code
    '''
    freq_keys = set(['P17', 'P131', 'P27', 'P150', 'P175', 'P577', 'P463', 'P527', 'P495', 'P361'])
    #time_dict = {'P577': 'publication date', 'P580': 'start time', 'P582': 'end time', 'P585': 'point in time', 'P569': 'date of birth', 'P570': 'date of death'}
    #freq_keys = set(time_dict.keys())
    long_tail_keys = set(rel2id.keys()) - freq_keys
    truth_dir = os.path.join(path, 'ref')

    if not os.path.exists(truth_dir):
        os.makedirs(truth_dir)

    fact_in_train_annotated = gen_train_facts(os.path.join(path, train_file), truth_dir)
    fact_in_train_distant = gen_train_facts(os.path.join(path, "train_distant.json"), truth_dir)

    truth = json.load(open(os.path.join(path, dev_file)))

    std = {}
    std_freq = {}
    std_long_tail = {}
    tot_evidences = 1
    titleset = set([])

    title2vectexSet = {}
    std_intra = {}
    std_inter = {}
    std_inter_long = {}
    for x in truth:
        title = x['title']
        titleset.add(title)

        vertexSet = x['vertexSet']
        title2vectexSet[title] = vertexSet

        for label in x['labels']:
            r = label['r']
            h_idx = label['h']
            t_idx = label['t']
            h_sent_set = [x['sent_id'] for x in vertexSet[h_idx]]
            t_sent_set = [x['sent_id'] for x in vertexSet[t_idx]]
            
            std[(title, r, h_idx, t_idx)] = set(label['evidence'])
            tot_evidences += len(label['evidence'])
            if findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set),len(t_sent_set) )==0:
                std_intra[(title, r, h_idx, t_idx)] = set(label['evidence'])
            if 1 <= findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set),len(t_sent_set) )  :
                std_inter[(title, r, h_idx, t_idx)] = set(label['evidence'])
            if 5 < findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set),len(t_sent_set) )  :
                std_inter_long[(title, r, h_idx, t_idx)] = set(label['evidence'])
            if r in freq_keys:
                std_freq[(title, r, h_idx, t_idx)] = set(label['evidence'])
            if r in long_tail_keys:
                std_long_tail[(title, r, h_idx, t_idx)] = set(label['evidence'])
                
    tot_relations = len(std)
    tot_relations_freq = len(std_freq)
    tot_relations_long_tail = len(std_long_tail)
    tot_relations_intra = len(std_intra)
    tot_relations_inter = len(std_inter)
    tot_relations_inter_long = len(std_inter_long)
    
    tmp.sort(key=lambda x: (x['title'], x['h_idx'], x['t_idx'], x['r']))
    if len(tmp) > 1:
        submission_answer = [tmp[0]]
        for i in range(1, len(tmp)):
            x = tmp[i]
            y = tmp[i - 1]
            if (x['title'], x['h_idx'], x['t_idx'], x['r']) != (y['title'], y['h_idx'], y['t_idx'], y['r']):
                submission_answer.append(tmp[i]) 
    else: 
        submission_answer = []
    submission_answer_freq = []
    submission_answer_long_tail =[] 

    submission_answer_freq = [x for x in submission_answer if x['r'] in freq_keys]
    submission_answer_long_tail = [x for x in submission_answer if x['r'] in long_tail_keys]
    submission_answer_intra = []
    submission_answer_inter = []
    submission_answer_inter_long = []
    for i in range(len(submission_answer)):
        vertexSet = title2vectexSet[submission_answer[i]['title']] 
        if title not in title2vectexSet:
            print(title)
            continue
        h_sent_set = [x['sent_id'] for x in vertexSet[submission_answer[i]['h_idx']]]
        t_sent_set = [x['sent_id'] for x in vertexSet[submission_answer[i]['t_idx']]]
        if findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set),len(t_sent_set) )==0:
            submission_answer_intra.append(submission_answer[i])
        if 1<= findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set),len(t_sent_set))  :
            submission_answer_inter.append(submission_answer[i])
        if 5 < findSmallestDifference(h_sent_set, t_sent_set, len(h_sent_set),len(t_sent_set)) :
            submission_answer_inter_long.append(submission_answer[i])

    correct_re = 0
    correct_re_freq = 0
    correct_re_long_tail = 0
    correct_re_intra = 0
    correct_re_inter = 0
    correct_re_inter_long = 0
    correct_evidence = 0
    pred_evi = 0

    correct_in_train_annotated = 0
    correct_in_train_distant = 0
    titleset2 = set([])
    for x in submission_answer:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if 'evidence' in x:
            evi = set(x['evidence'])
        else:
            evi = set([])
        pred_evi += len(evi)

        if (title, r, h_idx, t_idx) in std:
            correct_re += 1
            stdevi = std[(title, r, h_idx, t_idx)]
            correct_evidence += len(stdevi & evi)
            in_train_annotated = in_train_distant = False
            for n1 in vertexSet[h_idx]:
                for n2 in vertexSet[t_idx]:
                    if (n1['name'], n2['name'], r) in fact_in_train_annotated:
                        in_train_annotated = True
                    if (n1['name'], n2['name'], r) in fact_in_train_distant:
                        in_train_distant = True

            if in_train_annotated:
                correct_in_train_annotated += 1
            if in_train_distant:
                correct_in_train_distant += 1
    for x in submission_answer_freq:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if (title, r, h_idx, t_idx) in std_freq:
            correct_re_freq += 1
    for x in submission_answer_long_tail:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if (title, r, h_idx, t_idx) in std_long_tail:
            correct_re_long_tail += 1

    for x in submission_answer_intra:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if (title, r, h_idx, t_idx) in std_intra:
            correct_re_intra += 1
    for x in submission_answer_inter:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if (title, r, h_idx, t_idx) in std_inter:
            correct_re_inter += 1

 
    for x in submission_answer_inter_long:
        title = x['title']
        h_idx = x['h_idx']
        t_idx = x['t_idx']
        r = x['r']
        titleset2.add(title)
        if title not in title2vectexSet:
            continue
        vertexSet = title2vectexSet[title]

        if (title, r, h_idx, t_idx) in std_inter_long:
            correct_re_inter_long += 1


    if len(submission_answer)>0:        
        re_p = 1.0 * correct_re / len(submission_answer)
    else:
        re_p = 0
    re_r = 1.0 * correct_re / tot_relations
    if re_p + re_r == 0:
        re_f1 = 0
    else:
        re_f1 = 2.0 * re_p * re_r / (re_p + re_r)

    if len(submission_answer_freq)>0:        
        re_p_freq = 1.0 * correct_re_freq / len(submission_answer_freq)
    else:
        re_p_freq = 0
    
    re_r_freq = 1.0 * correct_re_freq / tot_relations_freq
    if re_p_freq + re_r_freq == 0:
        re_f1_freq = 0
    else:
        re_f1_freq = 2.0 * re_p_freq * re_r_freq / (re_p_freq + re_r_freq)
    if len(submission_answer_long_tail)>0:        
        re_p_long_tail = 1.0 * correct_re_long_tail / len(submission_answer_long_tail)
    else:
        re_p_long_tail = 0
    
    re_r_long_tail = 1.0 * correct_re_long_tail / tot_relations_long_tail
    if re_p_long_tail + re_r_long_tail == 0:
        re_f1_long_tail = 0
    else:
        re_f1_long_tail = 2.0 * re_p_long_tail * re_r_long_tail / (re_p_long_tail + re_r_long_tail)

    if len(submission_answer_intra)>0:        
        re_p_intra = 1.0 * correct_re_intra / len(submission_answer_intra)
    else:
        re_p_intra = 0
    
    re_r_intra = 1.0 * correct_re_intra / tot_relations_intra
    if re_p_intra + re_r_intra == 0:
        re_f1_intra = 0
    else:
        re_f1_intra = 2.0 * re_p_intra * re_r_intra / (re_p_intra + re_r_intra)

    if len(submission_answer_inter)>0:        
        re_p_inter = 1.0 * correct_re_inter / len(submission_answer_inter)
    else:
        re_p_inter = 0
    re_r_inter = 1.0 * correct_re_inter / tot_relations_inter
    if re_p_inter + re_r_inter == 0:
        re_f1_inter = 0
    else:
        re_f1_inter = 2.0 * re_p_inter * re_r_inter / (re_p_inter + re_r_inter)



    evi_p = 1.0 * correct_evidence / pred_evi if pred_evi > 0 else 0
    evi_r = 1.0 * correct_evidence / tot_evidences
    if evi_p + evi_r == 0:
        evi_f1 = 0
    else:
        evi_f1 = 2.0 * evi_p * evi_r / (evi_p + evi_r)

    re_p_ignore_train_annotated = 1.0 * (correct_re - correct_in_train_annotated) / (len(submission_answer) - correct_in_train_annotated + 1e-5)
    re_p_ignore_train = 1.0 * (correct_re - correct_in_train_distant) / (len(submission_answer) - correct_in_train_distant + 1e-5)

    if re_p_ignore_train_annotated + re_r == 0:
        re_f1_ignore_train_annotated = 0
    else:
        re_f1_ignore_train_annotated = 2.0 * re_p_ignore_train_annotated * re_r / (re_p_ignore_train_annotated + re_r)

    if re_p_ignore_train + re_r == 0:
        re_f1_ignore_train = 0
    else:
        re_f1_ignore_train = 2.0 * re_p_ignore_train * re_r / (re_p_ignore_train + re_r)

    return re_f1, evi_f1, re_f1_ignore_train_annotated, re_f1_ignore_train, re_p, re_r, re_f1_freq, re_f1_long_tail, re_f1_intra, re_f1_inter, re_p_freq, re_r_freq, re_p_long_tail, re_r_long_tail




def score_predictions_by_class(data, predictions):
    ground_truth = map_ground_truth(data)
    predictions_by_doc = []
    for i,example in enumerate(data):
        title = example['title']
        sent_lens = [len(x) for x in example['sents']]
        cum_sum = [0] + list(accumulate(sent_lens))
        entity_global_start = []
        ent_sents = []
        for entity in example['vertexSet']:
            g_start = []
            entity_s = [x['sent_id'] for x in entity]
            for mention in entity:
                mention['global_pos'] = [cum_sum[mention['sent_id']] + mention['pos'][0], cum_sum[mention['sent_id']] + mention['pos'][0]]
                mention['global_start'] = (cum_sum[mention['sent_id']] + mention['pos'][0])*100/cum_sum[-1]
                g_start.append(cum_sum[mention['sent_id']] + mention['pos'][0])
                #g_start.append((cum_sum[mention['sent_id']] + mention['pos'][0])*100/cum_sum[-1])
            entity_global_start.append(g_start)
            ent_sents.append(entity_s)
        example['e_starts'] = entity_global_start
        assert len(entity_global_start) == len(example['vertexSet'])
        example['ent_sents'] = ent_sents
        ex_preds = [(x['h_idx'], x['t_idx'], x['r']) for x in predictions[i]]
        predictions_by_doc.append(ex_preds)

    all_scores, scores_per_type = score(ground_truth, predictions_by_doc, type_idx=2, print_results=True)
    return scores_per_type

def score_predictions_by_class_bio(rels, predictions):
    ground_truth = rels
    predictions_by_doc = []
    for i,example in enumerate(predictions):
        ex_preds = [(x['h_idx'], x['t_idx'], x['r']) for x in predictions[i]]
        predictions_by_doc.append(ex_preds)
    print(predictions_by_doc[1])
    print(ground_truth[1])
    all_scores, scores_per_type = score_bio(ground_truth, predictions_by_doc, type_idx=2, print_results=True)
    return scores_per_type