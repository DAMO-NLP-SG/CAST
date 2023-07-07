import torch
import torch.nn as nn
from transformers import AutoModel
from opt_einsum import contract
from long_seq import process_long_input
from losses import ATLoss, NCRLoss


class DocREModel(nn.Module):
    def __init__(self, config, model_name, model=None, cache_dir=None, emb_size=768, block_size=64, num_labels=-1):
        super().__init__()
        self.config = config
        
        if model is not None:
            self.model = model
        else:
            self.model =  AutoModel.from_pretrained(model_name, config=config)
        self.hidden_size = config.hidden_size
        self.loss_fnt = ATLoss()

        self.head_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.tail_extractor = nn.Linear(2 * config.hidden_size, emb_size)
        self.bilinear = nn.Linear(emb_size * block_size, config.num_labels+2, bias=False)
        #self.bilinear = nn.Linear(emb_size * block_size, 15, bias=False)
        self.emb_size = emb_size
        self.block_size = block_size
        self.num_labels = num_labels

    def encode(self, input_ids, attention_mask):
        config = self.config
        if config.transformer_type == "bert":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id]
        elif config.transformer_type == "roberta":
            start_tokens = [config.cls_token_id]
            end_tokens = [config.sep_token_id, config.sep_token_id]
        sequence_output, attention = process_long_input(self.model, input_ids, attention_mask, start_tokens, end_tokens)
        return sequence_output, attention

    def get_hrt(self, sequence_output, attention, entity_pos, hts):
        offset = 1 if self.config.transformer_type in ["bert", "roberta"] else 0
        n, h, _, c = attention.size()
        hss, tss, rss = [], [], []
        for i in range(len(entity_pos)):
            entity_embs, entity_atts = [], []
            for e in entity_pos[i]:
                if len(e) > 1:
                    e_emb, e_att = [], []
                    for start, end in e:
                        if start + offset < c:
                            # In case the entity mention is truncated due to limited max seq length.
                            e_emb.append(sequence_output[i, start + offset])
                            e_att.append(attention[i, :, start + offset])
                    if len(e_emb) > 0:
                        e_emb = torch.logsumexp(torch.stack(e_emb, dim=0), dim=0)
                        e_att = torch.stack(e_att, dim=0).mean(0)
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                else:
                    start, end = e[0]
                    if start + offset < c:
                        e_emb = sequence_output[i, start + offset]
                        e_att = attention[i, :, start + offset]
                    else:
                        e_emb = torch.zeros(self.config.hidden_size).to(sequence_output)
                        e_att = torch.zeros(h, c).to(attention)
                entity_embs.append(e_emb)
                entity_atts.append(e_att)

            entity_embs = torch.stack(entity_embs, dim=0)  # [n_e, d]
            entity_atts = torch.stack(entity_atts, dim=0)  # [n_e, h, seq_len]
            
            ht_i = torch.LongTensor(hts[i]).to(sequence_output.device)
            #print(ht_i.size())
            hs = torch.index_select(entity_embs, 0, ht_i[:, 0])
            ts = torch.index_select(entity_embs, 0, ht_i[:, 1])

            h_att = torch.index_select(entity_atts, 0, ht_i[:, 0])
            t_att = torch.index_select(entity_atts, 0, ht_i[:, 1])
            ht_att = (h_att * t_att).mean(1)
            ht_att = ht_att / (ht_att.sum(1, keepdim=True) + 1e-5)
            rs = contract("ld,rl->rd", sequence_output[i], ht_att)
            hss.append(hs)
            tss.append(ts)
            rss.append(rs)
        hss = torch.cat(hss, dim=0)
        tss = torch.cat(tss, dim=0)
        rss = torch.cat(rss, dim=0)
        return hss, rss, tss

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                entity_pos=None,
                hts=None,
                instance_mask=None,
                negative_masks=None,
                inference_th=1.0,
                ):

        sequence_output, attention = self.encode(input_ids, attention_mask)
        hs, rs, ts = self.get_hrt(sequence_output, attention, entity_pos, hts)

        hs = torch.tanh(self.head_extractor(torch.cat([hs, rs], dim=1)))
        ts = torch.tanh(self.tail_extractor(torch.cat([ts, rs], dim=1)))
        b1 = hs.view(-1, self.emb_size // self.block_size, self.block_size)
        b2 = ts.view(-1, self.emb_size // self.block_size, self.block_size)
        b1 = b1.unsqueeze(3)
        b2 = b2.unsqueeze(2)
        bl = (b1 * b2).view(-1, self.emb_size * self.block_size)
        
        '''        
        sequence_output_1, attention_1 = self.encode(input_ids, attention_mask)
        hs_1, rs_1, ts_1 = self.get_hrt(sequence_output, attention, entity_pos, hts)

        hs_1 = torch.tanh(self.head_extractor(torch.cat([hs_1, rs_1], dim=1)))
        ts_1 = torch.tanh(self.tail_extractor(torch.cat([ts_1, rs_1], dim=1)))
        b1_1 = hs_1.view(-1, self.emb_size // self.block_size, self.block_size)
        b2_1 = ts_1.view(-1, self.emb_size // self.block_size, self.block_size)
        b1_1 = b1_1.unsqueeze(3)
        b2_1 = b2_1.unsqueeze(2)
        bl_1 = (b1_1 * b2_1).view(-1, self.emb_size * self.block_size)
        '''
        
        if negative_masks is not None:
            negative_masks = [torch.tensor(x) for x in negative_masks]
            negative_masks = torch.cat(negative_masks, dim=0).to(logits)
            negative_masks = torch.where(negative_masks == 1 )
            bl = bl[negative_masks]
        
        
        logits = self.bilinear(bl)
        #logits_1 = self.bilinear(bl_1)
        
        if labels is not None:
            
            labels = [torch.tensor(label) for label in labels]
            labels = torch.cat(labels, dim=0).to(logits)
            #if negative_masks is not None:
            #    labels = labels[negative_masks]
            loss = self.loss_fnt(logits.float(), labels.float())
            return loss
        else: 
            output = (self.loss_fnt.get_label(logits, num_labels=self.num_labels, inference_th=inference_th), logits)
            return output
