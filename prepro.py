from tqdm import tqdm
import ujson as json
import unidecode

import numpy as np

#docred_rel2id = json.load(open('meta/rel2id.json', 'r'))

docred_rel2id = {'P1376': 79, 'P607': 27, 'P136': 73, 'P137': 63, 'P131': 2, 'P527': 11, 'P1412': 38, 'P206': 33, 'P205': 77, 'P449': 52, 'P127': 34, 'P123': 49, 'P86': 66, 'P840': 85, 'P355': 72, 'P737': 93, 'P740': 84, 'P190': 94, 'P576': 71, 'P749': 68, 'P112': 65, 'P118': 40, 'P17': 1, 'P19': 14, 'P3373': 19, 'P6': 42, 'P276': 44, 'P1001': 24, 'P580': 62, 'P582': 83, 'P585': 64, 'P463': 18, 'P676': 87, 'P674': 46, 'P264': 10, 'P108': 43, 'P102': 17, 'P25': 81, 'P27': 3, 'P26': 26, 'P20': 37, 'P22': 30, 'Na': 0, 'P807': 95, 'P800': 51, 'P279': 78, 'P1336': 88, 'P577': 5, 'P570': 8, 'P571': 15, 'P178': 36, 'P179': 55, 'P272': 75, 'P170': 35, 'P171': 80, 'P172': 76, 'P175': 6, 'P176': 67, 'P39': 91, 'P30': 21, 'P31': 60, 'P36': 70, 'P37': 58, 'P35': 54, 'P400': 31, 'P403': 61, 'P361': 12, 'P364': 74, 'P569': 7, 'P710': 41, 'P1344': 32, 'P488': 82, 'P241': 59, 'P162': 57, 'P161': 9, 'P166': 47, 'P40': 20, 'P1441': 23, 'P156': 45, 'P155': 39, 'P150': 4, 'P551': 90, 'P706': 56, 'P159': 29, 'P495': 13, 'P58': 53, 'P194': 48, 'P54': 16, 'P57': 28, 'P50': 22, 'P1366': 86, 'P1365': 92, 'P937': 69, 'P140': 50, 'P69': 25, 'P1198': 96, 'P1056': 89}
docred_rel2id['FP'] = 97
docred_rel2id['FN'] = 98
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

ENTITY_PAIR_TYPE_SET = set(
    [("Chemical", "Disease"), ("Chemical", "Gene"), ("Gene", "Disease")])
cdr_rel2id = {'1:NR:2': 0, '1:CID:2': 1}
gda_rel2id = {'1:NR:2': 0, '1:GDA:2': 1}


def chunks(l, n):
    res = []
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        res += [l[i:i + n]]
    return res




def map_index(chars, tokens):
    # position index mapping from character level offset to token level offset
    ind_map = {}
    i, k = 0, 0  # (character i to token k)
    len_char = len(chars)
    num_token = len(tokens)
    while k < num_token:
        if i < len_char and chars[i].strip() == "":
            ind_map[i] = k
            i += 1
            continue
        token = tokens[k]
        if token[:2] == "##":
            token = token[2:]
        if token[:1] == "Ġ":
            token = token[1:]

        # assume that unk is always one character in the input text.
        if token != chars[i:(i+len(token))]:
            ind_map[i] = k
            i += 1
            k += 1
        else:
            for _ in range(len(token)):
                ind_map[i] = k
                i += 1
            k += 1

    return ind_map

def read_chemdisgene(args, file_in, tokenizer, max_seq_length=1024, lower=True):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    pos, neg, pos_labels, neg_labels = {}, {}, {}, {}
    for pair in list(ENTITY_PAIR_TYPE_SET):
        pos[pair] = 0
        neg[pair] = 0
        pos_labels[pair] = 0
        neg_labels[pair] = 0
    ent_nums = 0
    rel_nums = 0
    max_len = 0
    features = []
    lengths = []
    s_lengths = []
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    padid = tokenizer.pad_token_id
    all_pos_rels = []
    cls_token_length = len(cls_token)
    print(cls_token, sep_token)
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    re_fre = np.zeros(len(ctd_rel2id))
    for idx, sample in tqdm(enumerate(data), desc="Example"):
        if "title" in sample and "abstract" in sample:
            text = sample["title"] + sample["abstract"]
            if lower == True:
                text = text.lower()
        else:
            text = sample["text"]
            if lower == True:
                text = text.lower()

        text = unidecode.unidecode(text)
        lengths.append(len(text.split()))
        s_lengths.append(len(text.split('.')))
        tokens = tokenizer.tokenize(text)
        tokens = [cls_token] + tokens + [sep_token]
        text = cls_token + " " + text + " " + sep_token

        ind_map = map_index(text, tokens)

        entities = sample['entity']
        entity_start, entity_end = [], []

        train_triple = {}
        if "relation" in sample:
            for label in sample['relation']:
                if label['type'] not in ctd_rel2id:
                    continue
                if 'evidence' not in label:
                    evidence = []
                else:
                    evidence = label['evidence']
                r = int(ctd_rel2id[label['type']])

                if (label['subj'], label['obj']) not in train_triple:
                    train_triple[(label['subj'], label['obj'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['subj'], label['obj'])].append(
                        {'relation': r, 'evidence': evidence})

        entity_pos = []
        entity_dict = {}
        entity2id = {}
        entity_type = {}
        eids = 0
        pos_rels = []
        offset = 0

        for e in entities:

            entity_type[e["id"]] = e["type"]
            if int(e["start"]) + cls_token_length in ind_map:
                startid = ind_map[int(e["start"]) + cls_token_length] + offset
                tokens = tokens[:startid] + ['*'] + tokens[startid:]
                offset += 1
            else:
                continue
                startid = 0


            if int(e["end"]) + cls_token_length in ind_map:
                endid = ind_map[int(e["end"]) + cls_token_length] + offset
                if ind_map[int(e["start"]) + cls_token_length] >= ind_map[int(e["end"]) + cls_token_length]:
                    endid += 1
                tokens = tokens[:endid] + ['*'] + tokens[endid:]
                endid += 1
                offset += 1
            else:
                continue
                endid = 0

            if startid >= endid:
                endid = startid + 1

            if e["id"] not in entity_dict:
                entity_dict[e["id"]] = [(startid, endid,)]
                entity2id[e["id"]] = eids
                eids += 1
                if e["id"] != "-":
                    ent_nums += 1
            else:
                entity_dict[e["id"]].append((startid, endid,))

        relations, hts = [], []
        for h, t in train_triple.keys():
            #if h not in entity2id or t not in entity2id or ((entity_type[h], entity_type[t]) not in ENTITY_PAIR_TYPE_SET):
            if h not in entity2id or t not in entity2id:
                continue
            relation = [0] * (len(ctd_rel2id) + 1)
            
            for mention in train_triple[h, t]:
                #if relation[mention["relation"] ] == 0:
                #    re_fre[mention["relation"]] += 1
                relation[mention["relation"]] = 1
                evidence = mention["evidence"]
                pos_rels.append((entity2id[h], entity2id[t] , id2rel_ctd[mention["relation"]]))
            relations.append(relation)
            hts.append([entity2id[h], entity2id[t]])

            rel_num = sum(relation)
            rel_nums += rel_num
            pos_labels[(entity_type[h], entity_type[t])] += rel_num
            pos[(entity_type[h], entity_type[t])] += 1
            pos_samples += 1

        for h in entity_dict.keys():
            for t in entity_dict.keys():
                #if (h != t) and ([entity2id[h], entity2id[t]] not in hts) and ((entity_type[h], entity_type[t]) in ENTITY_PAIR_TYPE_SET) and (h != "-") and (t != "-"):
                if (h != t) and ([entity2id[h], entity2id[t]] not in hts) and (h != "-") and (t != "-"):

                    if (entity_type[h], entity_type[t]) not in neg:
                        neg[(entity_type[h], entity_type[t])] = 1
                    else:
                        neg[(entity_type[h], entity_type[t])] += 1
                    
                    relation = [1] + [0] * (len(ctd_rel2id))
                    relations.append(relation)
                    hts.append([entity2id[h], entity2id[t]])
                    neg_samples += 1

        if len(tokens) > max_len:
            max_len = len(tokens)

        tokens = tokens[1:-1][:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)
        all_pos_rels.append(pos_rels)
        i_line += 1

        feature = {'input_ids': input_ids,
                'entity_pos': list(entity_dict.values()),
                'labels': relations,
                'hts': hts,
                'title': sample['docid'],
                }
        features.append(feature)

    print("# of documents {}.".format(i_line))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    re_fre = 1. * re_fre / (pos_samples + neg_samples)
    print(re_fre)
    print(max_len)
    print(pos)
    print(pos_labels)
    print(neg)
    print("# words per doc", sum(lengths) / len(lengths))
    print("# sents per doc", sum(s_lengths) / len(s_lengths))
    print("# ents per doc", 1. * ent_nums / i_line)
    print("# rels per doc", 1. * rel_nums / i_line)
    return features, all_pos_rels

def read_docred_backup(file_in, tokenizer, max_seq_length=1024):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    for sample in tqdm(data, desc="Example"):
        sents = []
        sent_map = []

        entities = sample['vertexSet']
        entity_start, entity_end = [], []
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))
        for i_s, sent in enumerate(sample['sents']):
            new_map = {}
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)

        train_triple = {}
        if "labels" in sample:
            for label in sample['labels']:
                evidence = label['evidence']
                r = int(docred_rel2id[label['r']])
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r, 'evidence': evidence}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r, 'evidence': evidence})

        entity_pos = []
        for e in entities:
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end,))

        relations, hts = [], []
        for h, t in train_triple.keys():
            relation = [0] * len(docred_rel2id)
            for mention in train_triple[h, t]:
                relation[mention["relation"]] = 1
                evidence = mention["evidence"]
            relations.append(relation)
            hts.append([h, t])
            pos_samples += 1

        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t and [h, t] not in hts:
                    relation = [0] + [0] * (len(docred_rel2id) - 1)
                    relations.append(relation)
                    hts.append([h, t])
                    neg_samples += 1

        assert len(relations) == len(entities) * (len(entities) - 1)

        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        i_line += 1
        feature = {'input_ids': input_ids,
                   'entity_pos': entity_pos,
                   'labels': relations,
                   'hts': hts,
                   'title': sample['title'],
                   }
        features.append(feature)

    print("# of documents {}.".format(i_line))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    return features

def read_docred(file_in, tokenizer, max_seq_length=1024, drop_prob = 0.0):
    i_line = 0
    pos_samples = 0
    neg_samples = 0
    features = []
    if file_in == "":
        return None
    with open(file_in, "r") as fh:
        data = json.load(fh)

    for sample in tqdm(data, desc="Example"):
        sents = []
        sent_map = []

        entities = sample['vertexSet']
        entity_start, entity_end = [], []
        for entity in entities:
            for mention in entity:
                sent_id = mention["sent_id"]
                pos = mention["pos"]
                entity_start.append((sent_id, pos[0],))
                entity_end.append((sent_id, pos[1] - 1,))
        for i_s, sent in enumerate(sample['sents']):
            new_map = {}
            for i_t, token in enumerate(sent):
                tokens_wordpiece = tokenizer.tokenize(token)
                if (i_s, i_t) in entity_start:
                    tokens_wordpiece = ["*"] + tokens_wordpiece
                if (i_s, i_t) in entity_end:
                    tokens_wordpiece = tokens_wordpiece + ["*"]
                new_map[i_t] = len(sents)
                sents.extend(tokens_wordpiece)
            new_map[i_t + 1] = len(sents)
            sent_map.append(new_map)

        train_triple = {}
        if "labels" in sample:
            for label in sample['labels']:
                r = int(docred_rel2id[label['r']])
                if (label['h'], label['t']) not in train_triple:
                    train_triple[(label['h'], label['t'])] = [
                        {'relation': r,}]
                else:
                    train_triple[(label['h'], label['t'])].append(
                        {'relation': r,})
        distant_triple = {}
        if ("distant_labels" in sample) and False:
            for ds_label in sample['distant_labels']:
                r = int(docred_rel2id[ds_label['r']])
                if (ds_label['h'], ds_label['t']) not in distant_triple:
                    distant_triple[(ds_label['h'], ds_label['t'])] = [
                        {'relation': r, }]
                else:
                    distant_triple[(ds_label['h'], ds_label['t'])].append(
                        {'relation': r,})
        entity_pos = []
        for e in entities:
            entity_pos.append([])
            for m in e:
                start = sent_map[m["sent_id"]][m["pos"][0]]
                end = sent_map[m["sent_id"]][m["pos"][1]]
                entity_pos[-1].append((start, end,))

        relations, hts = [], []
        train_keys = set(train_triple.keys())
        distant_keys = set(distant_triple.keys())
        pos_hts = list(train_keys.union(distant_keys))
        intersect = train_keys.intersection(distant_keys)
        pos_dist = train_keys - distant_keys
        dist_pos = distant_keys - train_keys

        
        for (h, t) in pos_hts:
            relation = [0.0] * len(docred_rel2id)
            #print((h,t))
            #False Negative
            if (h, t) in pos_dist :
                for mention in train_triple[(h, t)]:
                    relation[mention["relation"]] = 1
                    r = int(docred_rel2id['FN'])
                    relation[r] = 0
            elif (h,t) in intersect:
                for mention in train_triple[(h, t)]:
                    relation[mention["relation"]] = 1
                target_tps = [x['relation'] for x in train_triple[h, t]]
                for mention in distant_triple[h, t]:
                    if mention['relation'] in target_tps:
                        continue
                    #False Positive
                    else:
                        relation[mention["relation"]] += 0.0
                        relation[int(docred_rel2id['FP'])] = 0
            elif (h,t) in dist_pos:
                #False Positive
                for mention in distant_triple[(h, t)]:
                    relation[mention["relation"]] += 0.0
                    relation[int(docred_rel2id['FP'])] = 0
                r = int(docred_rel2id['FP'])
                relation[r] = 0
            relations.append(relation)
            hts.append([h, t])
            pos_samples += 1

            
            
        neg_num =  len(entities) * (len(entities) - 1)  - pos_samples
        
        
        for h in range(len(entities)):
            for t in range(len(entities)):
                if h != t and [h, t] not in hts:
                    rand_num = np.random.rand()
                    if rand_num > drop_prob:
                        relation = [0] + [0] * (len(docred_rel2id) - 1)
                        relations.append(relation)
                        hts.append([h, t])
                        neg_samples += 1
                    else:
                        continue


        sents = sents[:max_seq_length - 2]
        input_ids = tokenizer.convert_tokens_to_ids(sents)
        input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

        i_line += 1
        feature = {'input_ids': input_ids,
                   'entity_pos': entity_pos,
                   'labels': relations,
                   'hts': hts,
                   'title': sample['title'],
                   }
        features.append(feature)
    print("# of documents {}.".format(i_line))
    print("# of positive examples {}.".format(pos_samples))
    print("# of negative examples {}.".format(neg_samples))
    
    return features


def read_cdr(file_in, tokenizer, max_seq_length=1024):
    pmids = set()
    features = []
    maxlen = 0
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)):
            line = line.rstrip().split('\t')
            pmid = line[0]

            if pmid not in pmids:
                pmids.add(pmid)
                text = line[1]
                prs = chunks(line[2:], 17)

                ent2idx = {}
                train_triples = {}

                entity_pos = set()
                for p in prs:
                    es = list(map(int, p[8].split(':')))
                    ed = list(map(int, p[9].split(':')))
                    tpy = p[7]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                    es = list(map(int, p[14].split(':')))
                    ed = list(map(int, p[15].split(':')))
                    tpy = p[13]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                sents = [t.split(' ') for t in text.split('|')]
                new_sents = []
                sent_map = {}
                i_t = 0
                for sent in sents:
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        for start, end, tpy in list(entity_pos):
                            if i_t == start:
                                tokens_wordpiece = ["*"] + tokens_wordpiece
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["*"]
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        i_t += 1
                    sent_map[i_t] = len(new_sents)
                sents = new_sents

                entity_pos = []

                for p in prs:
                    if p[0] == "not_include":
                        continue
                    if p[1] == "L2R":
                        h_id, t_id = p[5], p[11]
                        h_start, t_start = p[8], p[14]
                        h_end, t_end = p[9], p[15]
                    else:
                        t_id, h_id = p[5], p[11]
                        t_start, h_start = p[8], p[14]
                        t_end, h_end = p[9], p[15]
                    h_start = map(int, h_start.split(':'))
                    h_end = map(int, h_end.split(':'))
                    t_start = map(int, t_start.split(':'))
                    t_end = map(int, t_end.split(':'))
                    h_start = [sent_map[idx] for idx in h_start]
                    h_end = [sent_map[idx] for idx in h_end]
                    t_start = [sent_map[idx] for idx in t_start]
                    t_end = [sent_map[idx] for idx in t_end]
                    if h_id not in ent2idx:
                        ent2idx[h_id] = len(ent2idx)
                        entity_pos.append(list(zip(h_start, h_end)))
                    if t_id not in ent2idx:
                        ent2idx[t_id] = len(ent2idx)
                        entity_pos.append(list(zip(t_start, t_end)))
                    h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                    r = cdr_rel2id[p[0]]
                    if (h_id, t_id) not in train_triples:
                        train_triples[(h_id, t_id)] = [{'relation': r}]
                    else:
                        train_triples[(h_id, t_id)].append({'relation': r})

                relations, hts = [], []
                for h, t in train_triples.keys():
                    relation = [0] * len(cdr_rel2id)
                    for mention in train_triples[h, t]:
                        relation[mention["relation"]] = 1
                    relations.append(relation)
                    hts.append([h, t])

            maxlen = max(maxlen, len(sents))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            if len(hts) > 0:
                feature = {'input_ids': input_ids,
                           'entity_pos': entity_pos,
                           'labels': relations,
                           'hts': hts,
                           'title': pmid,
                           }
                features.append(feature)
    print("Number of documents: {}.".format(len(features)))
    print("Max document length: {}.".format(maxlen))
    return features


def read_gda(file_in, tokenizer, max_seq_length=1024):
    pmids = set()
    features = []
    maxlen = 0
    with open(file_in, 'r') as infile:
        lines = infile.readlines()
        for i_l, line in enumerate(tqdm(lines)):
            line = line.rstrip().split('\t')
            pmid = line[0]

            if pmid not in pmids:
                pmids.add(pmid)
                text = line[1]
                prs = chunks(line[2:], 17)

                ent2idx = {}
                train_triples = {}

                entity_pos = set()
                for p in prs:
                    es = list(map(int, p[8].split(':')))
                    ed = list(map(int, p[9].split(':')))
                    tpy = p[7]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                    es = list(map(int, p[14].split(':')))
                    ed = list(map(int, p[15].split(':')))
                    tpy = p[13]
                    for start, end in zip(es, ed):
                        entity_pos.add((start, end, tpy))

                sents = [t.split(' ') for t in text.split('|')]
                new_sents = []
                sent_map = {}
                i_t = 0
                for sent in sents:
                    for token in sent:
                        tokens_wordpiece = tokenizer.tokenize(token)
                        for start, end, tpy in list(entity_pos):
                            if i_t == start:
                                tokens_wordpiece = ["*"] + tokens_wordpiece
                            if i_t + 1 == end:
                                tokens_wordpiece = tokens_wordpiece + ["*"]
                        sent_map[i_t] = len(new_sents)
                        new_sents.extend(tokens_wordpiece)
                        i_t += 1
                    sent_map[i_t] = len(new_sents)
                sents = new_sents

                entity_pos = []

                for p in prs:
                    if p[0] == "not_include":
                        continue
                    if p[1] == "L2R":
                        h_id, t_id = p[5], p[11]
                        h_start, t_start = p[8], p[14]
                        h_end, t_end = p[9], p[15]
                    else:
                        t_id, h_id = p[5], p[11]
                        t_start, h_start = p[8], p[14]
                        t_end, h_end = p[9], p[15]
                    h_start = map(int, h_start.split(':'))
                    h_end = map(int, h_end.split(':'))
                    t_start = map(int, t_start.split(':'))
                    t_end = map(int, t_end.split(':'))
                    h_start = [sent_map[idx] for idx in h_start]
                    h_end = [sent_map[idx] for idx in h_end]
                    t_start = [sent_map[idx] for idx in t_start]
                    t_end = [sent_map[idx] for idx in t_end]
                    if h_id not in ent2idx:
                        ent2idx[h_id] = len(ent2idx)
                        entity_pos.append(list(zip(h_start, h_end)))
                    if t_id not in ent2idx:
                        ent2idx[t_id] = len(ent2idx)
                        entity_pos.append(list(zip(t_start, t_end)))
                    h_id, t_id = ent2idx[h_id], ent2idx[t_id]

                    r = gda_rel2id[p[0]]
                    if (h_id, t_id) not in train_triples:
                        train_triples[(h_id, t_id)] = [{'relation': r}]
                    else:
                        train_triples[(h_id, t_id)].append({'relation': r})

                relations, hts = [], []
                for h, t in train_triples.keys():
                    relation = [0] * len(gda_rel2id)
                    for mention in train_triples[h, t]:
                        relation[mention["relation"]] = 1
                    relations.append(relation)
                    hts.append([h, t])

            maxlen = max(maxlen, len(sents))
            sents = sents[:max_seq_length - 2]
            input_ids = tokenizer.convert_tokens_to_ids(sents)
            input_ids = tokenizer.build_inputs_with_special_tokens(input_ids)

            if len(hts) > 0:
                feature = {'input_ids': input_ids,
                           'entity_pos': entity_pos,
                           'labels': relations,
                           'hts': hts,
                           'title': pmid,
                           }
                features.append(feature)
    print("Number of documents: {}.".format(len(features)))
    print("Max document length: {}.".format(maxlen))
    return features
