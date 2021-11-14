
import torch
from torch import nn
import pytorch_lightning as pl
from typing import Any
from transformers import AutoConfig, AutoModel,AdamW, get_linear_schedule_with_warmup

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

class RbertClassifier(pl.LightningModule) :
    def __init__(self, model_name_or_path: str,hidden_size, num_labels, **kwargs ) -> None:
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name_or_path)
        self.model_base = AutoModel.from_pretrained(model_name_or_path)
        self.num_labels = num_labels
        self.hidden_size = hidden_size
        self.entity_layer = FCLayer(hidden_size, hidden_size)
        self.classify_layer = FCLayer(hidden_size, hidden_size)
        self.label_layer = FCLayer(3*hidden_size, hidden_size, use_activation=False)


    def forward(self,attention_mask, e1_mask, e2_mask, input_ids, token_type_ids):
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
        outputs = {
            "logits": logits,
            "hidden_state": outputs_model_base.last_hidden_state,
            "attention": outputs_model_base.attentions,
        }

        return outputs

    @staticmethod
    def entity_average(hidden_output, e_mask):
        e_mask_unsquezze = e_mask.unsqueeze(1)
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)
        sum_vector = torch.bmm(e_mask_unsquezze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()
        return avg_vector

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()

        labels = batch["labels"]

        return {"loss": val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):
        if self.hparams.task_name == "mnli":
            for i, output in enumerate(outputs):
                # matched or mismatched
                split = self.hparams.eval_splits[i].split("_")[-1]
                preds = torch.cat([x["preds"] for x in output]).detach().cpu().numpy()
                labels = torch.cat([x["labels"] for x in output]).detach().cpu().numpy()
                loss = torch.stack([x["loss"] for x in output]).mean()
                self.log(f"val_loss_{split}", loss, prog_bar=True)
                split_metrics = {
                    f"{k}_{split}": v for k, v in self.metric.compute(predictions=preds, references=labels).items()
                }
                self.log_dict(split_metrics, prog_bar=True)
            return loss

        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)
        return loss

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.train_dataloader()

        # Calculate total steps
        tb_size = self.hparams.train_batch_size * max(1, self.trainer.gpus)
        ab_size = self.trainer.accumulate_grad_batches * float(self.trainer.max_epochs)
        self.total_steps = (len(train_loader.dataset) // tb_size) // ab_size

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]