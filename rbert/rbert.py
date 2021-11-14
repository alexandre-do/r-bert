from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from datasets import Dataset, DatasetDict

# from datasets import Dataset, DatasetDict
from saola import FILES, TRN, VAL, MetricHolder, Registries, logger
from saola.sparam import read_cfg
from saola.std_input_output import ClassificationOutput, TextInput
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    BertModel,
    BertPreTrainedModel,
    DataCollatorWithPadding,
    PretrainedConfig,
)

from saola_transformers import rbert_module
from saola_transformers.base import (
    SaolaLightningModule,
    SaolaTrainer,
    _to_python_types,
    to_np,
)
from saola_transformers.data import BaseDataModule_, _get_random_elements, _num_workers
from saola_transformers.schema import RBERTAppCfg

LABEL, LR = "labels", "lr"
BE1, EE1, BE2, EE2 = "<e1>", "</e1>", "<e2>", "</e2>"
START, END, TAG, ID = "start", "end", "tag", "id"
SRC, DST = "src", "dest"
SEP_TOKEN, CLS_TOKEN = "[SEP]", "[CLS]"
LABEL_NO_REL = "no_relation"


@rbert_module.register_train
def train(cfg: RBERTAppCfg, **kwargs):
    model_path = cfg.model.model_name_or_path
    # Load the retrained tokenizer
    special_toks = list(set(cfg.model.special_toks + [BE1, EE1, BE2, EE2]))
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        additional_special_tokens=special_toks,
    )
    dm = RBERTDataModule(cfg, tokenizer)
    dm.setup("fit")
    list_labels = dm.get_relations()
    config_rbert = PretrainedConfig.from_pretrained(
        model_path,
        return_dict=True,
        num_labels=len(list_labels),
        id2label={str(i): label for i, label in enumerate(list_labels)},
        label2id={label: i for i, label in enumerate(list_labels)},
    )
    rbert_model = RBERTModel(config_rbert)
    rbert_module = RBERTModule(model=rbert_model, tokenizer=tokenizer, cfg=cfg)
    trainer = SaolaTrainer.from_cfg(cfg)
    trainer.fit(rbert_module, dm)
    return trainer.saola_metrics


@rbert_module.register_predict
def predict(outpath: Path, model_path: Path, *, cfg_kwargs: Dict[str, Any] = {}):
    if not model_path.exists():
        raise ValueError("Invalid Path for model_path")
    cfg = read_cfg(RBERTAppCfg, model_path / "cfg.yaml")
    for k, v in cfg_kwargs.items():
        cfg.set(k, v)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    config_rbert = PretrainedConfig.from_pretrained(model_path)
    model_rbert = RBERTModel.from_pretrained(model_path, config=config_rbert)
    model = RBERTModule(
        model=model_rbert,
        tokenizer=tokenizer,
        cfg=cfg,
    )
    dm = RBERTDataModule(cfg, tokenizer)
    dm.setup("predict")
    dataset = dm.dataset

    def _predict(split):
        logger.pretty(f"[magenta]Making prediction on {split} split[/]")
        has_gpu = torch.cuda.is_available()
        trainer = pl.Trainer(logger=False, gpus=cfg.run.gpus if has_gpu else None)
        dl = dm._create_dl(
            dataset[split],
            cfg.training.eval_batch_size,
            _num_workers(len(dataset[split])),
        )
        trainer.test(model, dl, verbose=False)  # type: ignore
        return model.holder.tst_logits, model.holder.tst_labels

    ls = []
    for split in ["train", "validation", "test"]:
        if split not in dataset:
            continue
        logits, labels = _predict(split)
        id_labels = np.where(labels == 1)[1]
        id_max_logits = np.argmax(logits, axis=1)
        text_with_tags = dm.rbert_dataset[split]["texts"]
        for i in range(len(logits)):
            ls.append(
                [
                    split,
                    text_with_tags[i],
                    dm.id2relation[str(id_labels[i])],
                    dm.id2relation[str(id_max_logits[i])],
                    *logits[i],
                ]
            )
        if metrics := model.holder.val_metrics:
            logger.print_metrics(_to_python_types(metrics.main))
            logger.pretty_table(metrics.per_target)
    df = pd.DataFrame(ls)
    df.columns = ["Subset", "Text", "True label", "Predict label"] + [
        "logit_label_{}".format(dm.id2relation.get(str(i))) for i in range(model_rbert.num_labels)
    ]
    outpath = model_path / "my_predict" if outpath == Path("my-pred") else outpath
    outpath.mkdir(exist_ok=True, parents=True)
    df.to_csv(outpath / "predict.csv", index=None, sep=";")
    logger.pretty(f"Output to {outpath}")


class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super().__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class RBERTModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model_base = BertModel(config=config)
        num_labels, hidden_size = config.num_labels, config.hidden_size
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.cls_fc_layer = FCLayer(hidden_size, hidden_size)
        self.entity_fc_layer = FCLayer(hidden_size, hidden_size)
        self.label_classifier = FCLayer(hidden_size * 3, num_labels, use_activation=False)

    @staticmethod
    def entity_average(hidden_output, e_mask):
        e_mask_unsquezze = e_mask.unsqueeze(1)
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)
        sum_vector = torch.bmm(e_mask_unsquezze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()
        return avg_vector

    def forward(self, attention_mask, e1_mask, e2_mask, input_ids, token_type_ids):
        # output : last_hidden_state, pooler_output, hidden_states, attention
        outputs_model_base = self.model_base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True,
            output_attentions=True,
        )
        last_hidden_state = outputs_model_base.last_hidden_state
        pooler_output = outputs_model_base.pooler_output  # CSL

        # Average of tokens in e1_mask and e2_mask
        e1 = self.entity_average(last_hidden_state, e1_mask)
        e2 = self.entity_average(last_hidden_state, e2_mask)

        # Dropout -> tanh -> fc_layer
        pooler_output = self.cls_fc_layer(pooler_output)
        e1 = self.entity_fc_layer(e1)
        e2 = self.entity_fc_layer(e2)

        # Concat -> fc_layer
        concat_h = torch.cat([pooler_output, e1, e2], dim=-1)
        logits = self.label_classifier(concat_h)

        # Add hidden states and attention, if there are
        # logits, hidden_state, attention
        outputs = {
            "logits": logits,
            "hidden_state": outputs_model_base.last_hidden_state,
            "attention": outputs_model_base.attentions,
        }

        return outputs


class RBERTModule(SaolaLightningModule):
    def __init__(self, model, tokenizer, cfg):
        super().__init__(cfg, model, tokenizer)
        # Load BERT pretrained model
        self.model = model
        # Loss function
        self.loss = Registries.losses.resolve(
            cfg.model.loss, labels=["label_{}".format(i) for i in range(self.model.num_labels)]
        )
        self.holder = _PredictionHolder()
        self.metric_fn = Registries.metrics.resolve(
            cfg.training.metrics, ["label_{}".format(i) for i in range(self.model.num_labels)]
        )

    def forward(self, attention_mask, e1_mask, e2_mask, input_ids, labels, token_type_ids):
        # output : loss(if label is given), last_hidden_state, hidden_states, attentions
        outputs = self.model(attention_mask, e1_mask, e2_mask, input_ids, token_type_ids)
        if labels is not None:
            loss_fn = self.loss.to(self.device) if hasattr(self.loss, "to") else self.loss
            loss = loss_fn(outputs.get("logits"), labels)
            # loss, logits, hidden_states, attention
            outputs.update({"loss": loss})
        return outputs

    def training_step(self, batch, batch_idx):
        # attention_mask, e1_mask, e2_mask, input_ids, labels, token_type_ids = batch
        output = self(**batch)
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        metrics = {TRN.LOSS: output.get("loss"), TRN.LR: lr}
        self.log_dict(metrics)
        return metrics.get(TRN.LOSS)

    def validation_step(self, batch, batch_idx):
        # attention_mask, e1_mask, e2_mask, input_ids, labels, token_type_ids = batch
        labels = batch[LABEL]
        outputs = self(**batch)

        return {VAL.LOSS: outputs.get("loss"), "logits": outputs.get("logits"), LABEL: labels}

    def validation_epoch_end(self, outputs):
        logits = torch.cat([x["logits"] for x in outputs]).detach()
        labels = torch.cat([x[LABEL] for x in outputs]).detach()
        loss = torch.stack([x[VAL.LOSS] for x in outputs]).mean()
        holder = self.holder
        holder.val_labels = to_np(labels)
        holder.val_logits = to_np(logits)

        if not self.trainer.running_sanity_check:
            metrics: MetricHolder = self.metric_fn(
                val_ys=holder.val_labels,
                val_ps=holder.val_logits,
                trn_ys=_empty_list_to_none(holder.trn_labels),
                trn_ps=_empty_list_to_none(holder.trn_logits),
                sanity_check=self.trainer.running_sanity_check,
            )
        else:
            metrics = MetricHolder()

        metrics.main[VAL.LOSS] = _to_python_types(loss)
        holder.val_metrics = metrics
        holder.trn_labels.clear()
        holder.trn_logits.clear()
        self.log_dict(metrics.main)

    def test_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        outputs = self(**batch)
        result = {"logits": outputs.get("logits")}
        if "labels" in batch:
            result.update({"loss": outputs.get("loss"), "labels": batch["labels"]})
        return result

    def test_epoch_end(self, outputs):
        holder = self.holder
        logits = torch.cat([x["logits"] for x in outputs]).detach()
        holder.tst_logits = to_np(logits)
        if "labels" in outputs[0]:
            labels = torch.cat([x["labels"] for x in outputs]).detach()
            holder.tst_labels = to_np(labels)
            holder.val_metrics = self.metric_fn(
                val_ys=holder.tst_labels,
                val_ps=holder.tst_logits,
                trn_ys=_empty_list_to_none(holder.trn_labels),
                trn_ps=_empty_list_to_none(holder.trn_logits),
            )


class RBERTDataModule(BaseDataModule_):
    loader_columns = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "e1_mask",
        "e2_mask",
        "labels",
    ]

    def __init__(self, cfg: RBERTAppCfg, tokenizer, **kwargs):
        super().__init__(cfg, tokenizer)

    def convert_to_rbert_format(self, data):
        text_col, ent_col, rel_col = (
            self.cfg.dataset.text_column,
            self.cfg.dataset.entity_column,
            self.cfg.dataset.relation_column,
        )
        text, d_ents, d_rels = (
            data[text_col],
            data[ent_col],
            data[rel_col],
        )

        max_window = self.cfg.dataset.window_size
        l_rels_allow = self.cfg.dataset.allowed_relation_types
        l_tagged_texts, l_labels = [], []
        # Dict of all allowed couples for each relation type
        d_ents_rel_allow = self._get_dict_entity_ids_by_allowed_relation_type(l_rels_allow, d_ents)
        # Dict of all couples with labels in datatset for each relation type
        d_ents_rel = self._get_dict_entity_ids_by_relation_type(d_rels, d_ents)

        for relation_type, l_all_couple in d_ents_rel_allow.items():
            # for relation type, find all couple having this relation in dataset
            l_couple_rel = d_ents_rel[relation_type] if relation_type in d_ents_rel else []
            # for relation type, find all couple no having this relation in dataset
            l_couple_no_rel = list(set(l_all_couple).difference(set(l_couple_rel)))

            # filter couple by max_window
            l_couple_rel_within_window = [
                couple
                for couple in l_couple_rel
                if (isinstance(couple[2], int)) and (couple[2] <= max_window)
            ]
            # filter couple by max_window
            l_couple_no_rel_within_window = [
                couple
                for couple in l_couple_no_rel
                if (isinstance(couple[2], int)) and (couple[2] <= max_window)
            ]

            # add the couples having no this relation
            l_labels += [LABEL_NO_REL] * len(l_couple_no_rel_within_window)
            for couple in l_couple_no_rel_within_window:
                id_src, id_dest = couple[0], couple[1]
                if isinstance(id_src, int) and isinstance(id_dest, int):
                    src = self._get_entity_by_id(d_ents, id_src)
                    dest = self._get_entity_by_id(d_ents, id_dest)
                    tagged_text = self.add_tags_entities_to_text(text, src, dest)
                    l_tagged_texts.append(tagged_text)

            # add the couples having this relation
            l_labels += [relation_type] * len(l_couple_rel_within_window)
            for couple in l_couple_rel_within_window:
                id_src, id_dest = couple[0], couple[1]
                if isinstance(id_src, int) and isinstance(id_dest, int):
                    src = self._get_entity_by_id(d_ents, id_src)
                    dest = self._get_entity_by_id(d_ents, id_dest)
                    tagged_text = self.add_tags_entities_to_text(text, src, dest)
                    l_tagged_texts.append(tagged_text)

        return {"texts": l_tagged_texts, "labels": l_labels}

    @staticmethod
    def add_tags_entities_to_text(text, src, dest):
        start_src, end_src = src.get(START), src.get(END)
        start_dest, end_dest = dest.get(START), dest.get(END)
        if (start_src is None) or (end_src is None) or (start_dest is None) or (end_dest is None):
            raise ValueError("not find index for ner in text")
        if start_src < start_dest:
            text = f"{text[: start_dest]}{BE2}{text[start_dest : end_dest]}{EE2}{text[end_dest :]}"
            text = f"{text[: start_src]}{BE1}{text[start_src : end_src]}{EE1}{text[end_src :]}"
        else:
            text = f"{text[: start_src]}{BE1}{text[start_src : end_src]}{EE1}{text[end_src :]}"
            text = f"{text[: start_dest]}{BE2}{text[start_dest : end_dest]}{EE2}{text[end_dest :]}"
        return text

    @staticmethod
    def _get_entity_by_id(list_ents, id):
        for ent in list_ents:
            if ent[ID] == id:
                return ent
        raise Exception(f"Entity with id = {id} not found")

    @staticmethod
    def _get_dict_entity_id_by_entity_type(list_ents):
        dict_ids_by_type = dict()
        dict_start_char = dict()
        for ent in list_ents:
            type_, id_, start_ = ent[TAG], ent[ID], ent[START]
            type_ = type_ if type_ is not None else "none_type"
            id_ = id_ if id_ is not None else "none_id"
            dict_start_char[id_] = start_
            if type_ in dict_ids_by_type:
                dict_ids_by_type[type_].append(id_)
            else:
                dict_ids_by_type[type_] = [id_]
        return dict_ids_by_type, dict_start_char

    def _get_dict_entity_ids_by_relation_type(self, dict_relations, dict_entities):
        dict_ids_by_relation = dict()
        for rel in dict_relations:
            type_, src_, dest_ = rel[TAG], rel[SRC], rel[DST]
            type_ = type_ if type_ is not None else "none_type"
            if isinstance(src_, int) and isinstance(dest_, int):
                dist_ = abs(
                    self._get_entity_by_id(dict_entities, src_)[START]
                    - self._get_entity_by_id(dict_entities, dest_)[START]
                )
            else:
                dist_ = "none_dist"
            if type_ in dict_ids_by_relation:
                dict_ids_by_relation[type_].append((src_, dest_, dist_))
            else:
                dict_ids_by_relation[type_] = [(src_, dest_, dist_)]
        return dict_ids_by_relation

    def _get_dict_entity_ids_by_allowed_relation_type(
        self, list_allowed_relation_types, list_entities
    ):
        d = dict()
        dict_id_type_relation, dict_start_id_entity = self._get_dict_entity_id_by_entity_type(
            list_entities
        )
        for relation in list_allowed_relation_types:
            relation_type, allowed_ents_types = relation.code, relation.allowed_ents_types
            l_src_dest_dist = []
            for (type_src, type_dest) in allowed_ents_types:
                if (type_src not in dict_id_type_relation) or (
                    type_dest not in dict_id_type_relation
                ):
                    continue
                ids_src, ids_dest = (
                    dict_id_type_relation[type_src],
                    dict_id_type_relation[type_dest],
                )
                l_src_dest_dist += [
                    (
                        id_src,
                        id_dest,
                        abs(dict_start_id_entity[id_src] - dict_start_id_entity[id_dest]),
                    )
                    for id_src in ids_src
                    for id_dest in ids_dest
                    if id_src != id_dest
                ]
            d[relation_type] = l_src_dest_dist
        return d

    @staticmethod
    def _flatten_dataset_of_list(dataset):
        dataset_flatten = dict()
        for k in list(dataset.features.keys()):
            if isinstance(dataset[k][0], list):
                dataset_flatten[k] = [
                    item_ for items_sample in dataset[k] for item_ in items_sample
                ]
        return Dataset.from_dict(dataset_flatten)

    def setup(self, stage):
        self.org_dataset = Registries.datasets.resolve(self.cfg.dataset)

        # Set up list of relation types
        self.setup_list_relations()
        # convert raw data into rbert data (text with tags e1, e2)
        self.rbert_dataset = self.org_dataset.map(
            self.convert_to_rbert_format,
            remove_columns=["my_entities", "my_relations", "my_text", "generated_row_id"],
        )
        # flatten if a sample is a list (when one text is transformed many text with NER tags, we'll flatten all texts)
        for split in self.rbert_dataset:
            self.rbert_dataset[split] = self._flatten_dataset_of_list(self.rbert_dataset[split])
        # convert rbert data into feature dataset
        self.dataset = DatasetDict(
            {split: self.setup_split(ds) for split, ds in self.rbert_dataset.items()}
        )
        self.num_workers = _num_workers(sum(self.dataset.num_rows.values()))

    def convert_to_features(self, batch):
        pad_token, pad_token_id, cls_token_id, sep_token_id, sequence_token_id = 0, 0, 0, 0, 0
        texts, labels = batch["texts"], batch["labels"]
        features = dict(
            input_ids=[], attention_mask=[], token_type_ids=[], e1_mask=[], e2_mask=[], labels=[]
        )

        # features = []
        for (text, label) in zip(texts, labels):
            label = int(self.relation2id[label])
            id_label = [0] * len(self.relations)
            id_label[label] = 1
            # Tokenize the text
            tokens = self.tokenizer.tokenize(text)
            # - 2 is number of special tokens CLS and SEP.
            tokens = tokens[: (self.cfg.training.max_seq_length - 2)]
            tokens = [CLS_TOKEN] + tokens + [SEP_TOKEN]
            if (
                (BE1 not in tokens)
                or (EE1 not in tokens)
                or (BE2 not in tokens)
                or (EE2 not in tokens)
            ):
                logger.warning(
                    f"Entity 1 is out of max sequences length : {self.cfg.training.max_seq_length} tokens - skip this sample"
                )
                continue
            e11_p, e12_p = tokens.index(BE1), tokens.index(EE1)
            e21_p, e22_p = tokens.index(BE2), tokens.index(EE2)

            tokens[e11_p], tokens[e12_p] = "$", "$"
            tokens[e21_p], tokens[e22_p] = "#", "#"
            # Input tokens sequence
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # Padding lenght
            padding_len = self.cfg.training.max_seq_length - len(input_ids)
            input_ids = input_ids + ([pad_token] * padding_len)
            # The mask
            attention_mask = [1] * len(tokens)
            attention_mask = attention_mask + [0] * padding_len
            # The tokens ID
            token_type_ids = (
                [cls_token_id]
                + [sequence_token_id] * (len(tokens) - 2)
                + [sep_token_id]
                + [pad_token_id] * padding_len
            )
            # mask sequence for entity1, entity2
            e1_mask, e2_mask = [0] * len(attention_mask), [0] * len(attention_mask)
            e1_mask[e11_p : (e12_p + 1)] = [1] * (e12_p + 1 - e11_p)
            e2_mask[e21_p : (e22_p + 1)] = [1] * (e22_p + 1 - e21_p)
            features["attention_mask"].append(attention_mask)
            features["e1_mask"].append(e1_mask)
            features["e2_mask"].append(e2_mask)
            features["input_ids"].append(input_ids)
            features["labels"].append(id_label)
            features["token_type_ids"].append(token_type_ids)
        return features

    def _create_dl(self, ds, bs, workers):
        return DataLoader(
            ds,
            batch_size=bs,
            num_workers=workers,
            pin_memory=True,
            # TODO: check pad_to_multiple_of arg for f16
            collate_fn=DataCollatorWithPadding(self.tokenizer),
        )

    def show_stats(self):
        logger.pretty_table(self._random_examples(), "Random examples")
        logger.pretty_table(self._count_relations(), "Relations stats")
        logger.pretty_table(
            self._get_text_stats([self.cfg.dataset.get_raw("text_column")]),
            "Text length quantiles (characters)",
        )

    def _count_relations(self):
        ls = []
        rbert_dataset, list_relations = self.rbert_dataset, self.relations
        for i, rel in enumerate(list_relations):
            row = {"Relation": rel}
            for split in ["train", "validation", "test"]:
                if split not in rbert_dataset:
                    continue
                row[f"{split}"] = rbert_dataset[split]["labels"].count(rel)
                ls.append(row)
        return ls

    def setup_list_relations(self):
        if not hasattr(self, "relations"):
            allowed_relations = self.cfg.dataset.allowed_relation_types
            relations = set([rel.code for rel in allowed_relations] + [LABEL_NO_REL])
            self.relations = relations
            self.id2relation = {str(i): rel for i, rel in enumerate(relations)}
            self.relation2id = {rel: str(i) for i, rel in enumerate(relations)}

    def get_relations(self):
        if not hasattr(self, "relations"):
            self.setup_list_relations(self)
        return self.relations

    def _random_examples(self, k=5):
        ds = self.rbert_dataset["train"]
        tpls = ["[cyan]{v}[/]", "[green]{v}[/]", "[magenta]{v}[/]"]
        examples = _get_random_elements(len(ds), k)
        ls = []
        for i in examples:
            text = ds[i]["texts"]
            label = ds[i]["labels"]
            ls.append(
                {
                    "Text": text,
                    "Label": tpls[int(self.relation2id[label]) % len(tpls)].format(v=label),
                }
            )
        return ls


@dataclass
class _PredictionHolder:
    trn_labels: List[np.array] = field(default_factory=list)
    trn_logits: List[np.array] = field(default_factory=list)
    val_labels: Optional[np.array] = None
    val_logits: Optional[np.array] = None
    tst_labels: Optional[np.array] = None
    tst_logits: Optional[np.array] = None
    tst_ids = None

    val_metrics: MetricHolder = MetricHolder()


@rbert_module.register_load_pretrained
def load_pretrained(model_path: Path, **kwargs):
    cfg = read_cfg(RBERTAppCfg, model_path / FILES.CFG)
    return PreTrainedRBERTModel.from_pretrained(model_path, cfg)


class PreTrainedRBERTModel:
    def __init__(self, model, tokenizer, cfg: RBERTAppCfg, config):
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.config = config

    @classmethod
    def from_pretrained(self, model_path: Path, cfg: RBERTAppCfg, **kwargs):
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        config = PretrainedConfig.from_pretrained(model_path)
        model = RBERTModel.from_pretrained(model_path, config=config)
        return PreTrainedRBERTModel(model, tokenizer, cfg=cfg, config=config)

    def predict(self, input: TextInput) -> ClassificationOutput:
        raise NotImplementedError

    def batch_predict(self, inputs: List[TextInput]) -> List[ClassificationOutput]:
        raise NotImplementedError

    def _get_allowed_couple_entity(self, list_entities):
        max_window = self.cfg.dataset.window_size
        l_rels_allow = self.cfg.dataset.allowed_relation_types
        # List of all allowed couples for each relation type
        list_allowed_couple_entity = RBERTDataModule(
            self.cfg, self.tokenizer
        )._get_dict_entity_ids_by_allowed_relation_type(l_rels_allow, list_entities)
        # List of all couples within the max_window
        l_couples, l_relations_allowed = [], []
        get_entity_by_id = RBERTDataModule._get_entity_by_id
        for key_relation, list_couples in list_allowed_couple_entity.items():
            l_relations_allowed += [
                key_relation for couple in list_couples if couple[2] <= max_window
            ]
            l_couples += [
                [
                    get_entity_by_id(list_entities, couple[0]),
                    get_entity_by_id(list_entities, couple[1]),
                ]
                for couple in list_couples
                if couple[2] <= max_window
            ]

        return l_couples, l_relations_allowed

    def __call__(self, text: List[str], list_entities: List[List]) -> List[List[Dict]]:
        # Apply the allowed relation types for inputs
        l_couples, l_relations_allowed = self._get_allowed_couple_entity(list_entities)
        # Convert raw text to text with entity tags
        add_tags_entities_to_text = RBERTDataModule.add_tags_entities_to_text
        # Case there is no allowed couple
        if len(l_couples) == 0:
            return []
        texts_with_tags_entity = [
            add_tags_entities_to_text(text, couples[0], couples[1]) for couples in l_couples
        ]
        # Tokenizer the texts and prepare inputs for model
        list_features = self.prepare_predict_data(texts_with_tags_entity)
        for k, v in list_features.items():
            list_features[k] = torch.tensor(v)
        # Predict the relations
        outs = self.model(**list_features)
        list_id_relations = outs["logits"].argmax(axis=1).tolist()
        l_relations_predicted = [
            self.config.id2label[id_relation] for id_relation in list_id_relations
        ]
        # return l_couples, l_relations_allowed, l_relations_predicted
        # Write the outputs
        results = []
        for relation_predicted, relation_allowed, couple in zip(
            l_relations_predicted, l_relations_allowed, l_couples
        ):
            results.append(
                dict(
                    src=couple[0][ID],
                    dest=couple[1][ID],
                    tag=relation_predicted,
                    rel_allowed=relation_allowed,
                )
            )
        return results

    def prepare_predict_data(self, texts_with_tags: List[str]):
        pad_token, pad_token_id, cls_token_id, sep_token_id, sequence_token_id = 0, 0, 0, 0, 0
        features = dict(input_ids=[], attention_mask=[], token_type_ids=[], e1_mask=[], e2_mask=[])
        # features = []
        for text in texts_with_tags:
            # Tokenize the text
            tokens = self.tokenizer.tokenize(text)
            # - 2 is number of special tokens CLS and SEP.
            tokens = tokens[: (self.cfg.training.max_seq_length - 2)]
            tokens = [CLS_TOKEN] + tokens + [SEP_TOKEN]
            try:
                e11_p, e12_p = tokens.index(BE1), tokens.index(EE1)
                e21_p, e22_p = tokens.index(BE2), tokens.index(EE2)
            except Exception:
                continue
            tokens[e11_p], tokens[e12_p] = "$", "$"
            tokens[e21_p], tokens[e22_p] = "#", "#"
            # Input tokens sequence
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            # Padding lenght
            padding_len = self.cfg.training.max_seq_length - len(input_ids)
            input_ids = input_ids + ([pad_token] * padding_len)
            # The mask
            attention_mask = [1] * len(tokens)
            attention_mask = attention_mask + [0] * padding_len
            # The tokens ID
            token_type_ids = (
                [cls_token_id]
                + [sequence_token_id] * (len(tokens) - 2)
                + [sep_token_id]
                + [pad_token_id] * padding_len
            )
            # mask sequence for entity1, entity2
            e1_mask, e2_mask = [0] * len(attention_mask), [0] * len(attention_mask)
            e1_mask[e11_p : (e12_p + 1)] = [1] * (e12_p + 1 - e11_p)
            e2_mask[e21_p : (e22_p + 1)] = [1] * (e22_p + 1 - e21_p)
            features["attention_mask"].append(attention_mask)
            features["e1_mask"].append(e1_mask)
            features["e2_mask"].append(e2_mask)
            features["input_ids"].append(input_ids)
            features["token_type_ids"].append(token_type_ids)
        return features


def _empty_list_to_none(x):
    return None if len(x) == 0 else np.vstack(x)
