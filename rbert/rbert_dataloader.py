from pytorch_lightning import LightningDataModule, DataLoader
from transformers import AutoTokenizer
from typing import Optional

LABEL, LR = "labels", "lr"
BE1, EE1, BE2, EE2 = "<e1>", "</e1>", "<e2>", "</e2>"
START, END, TAG, ID = "start", "end", "tag", "id"
SRC, DST = "src", "dest"
SEP_TOKEN, CLS_TOKEN = "[SEP]", "[CLS]"
LABEL_NO_REL = "no_relation"

class RbertDataModule(LightningDataModule):
    loader_columns = [
        "input_ids",
        "attention_mask",
        "token_type_ids",
        "e1_mask",
        "e2_mask",
        "labels",
    ]
    def __init__(self, model_name_or_path:str, max_seq_len:int =512, batch_size_train :int =32, batch_size_valid:int=32, text_column:str, label_column:str):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.batch_size_train = batch_size_train
        self.batch_size_valid = batch_size_valid
        self.text_column = text_column
        self.label_column = label_column
        self.model_name_or_path = model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)

    # To load, or donwload the data
    def prepare_data(self) -> None:
        self.datasets = ''
        return super().prepare_data()

    # 
    def convert_batch_to_features(self, batch):
        features = batch
        # tokenizering the data

        # add others features 

        # rename the features 
        
        return features


    # To prepare the split 
    def setup(self, stage: str):    
        for split in self.datasets.keys():
            self.datasets[split] = self.datasets[split].map(self.convert_batch_to_features, batch=True, remove_columns=['labels'])
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)
        self.splits_validation = [name for name in self.datasets.keys() if 'valid' in name]
        return super().setup(stage=stage)

    def train_dataloader(self):
        return DataLoader(self.datasets['train'], batch_size = self.batch_size_train)

    def val_dataloader(self):
        if len(self.splits_validation) ==1:
            return DataLoader(self.datasets['validation'], batch_size = self.batch_size_valid)
        else:
            return [DataLoader(self.datasets[split], batch_size = self.batch_size_valid) for split in self.splits_validation]

    def test_dataloader(self) :
        if len(self.splits_validation) ==1:
            return DataLoader(self.datasets['test'], batch_size = self.batch_size_valid)
        else:
            return [DataLoader(self.datasets[split], batch_size = self.batch_size_valid) for split in self.splits_validation]
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

