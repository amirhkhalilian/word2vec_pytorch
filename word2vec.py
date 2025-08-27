import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class Word2Vec(pl.LightningModule):
    def __init__(self,
                 vocab_size,
                 embed_dim = 50, # typical value is 300
                 lr = 1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.in_embed = nn.Embedding(vocab_size, embed_dim, max_norm=None)
        self.out_embed = nn.Embedding(vocab_size, embed_dim, max_norm=None)

    def forward(self, center, context, negatives):
        center_emb = self.in_embed(center)
        context_emb = self.out_embed(context)
        neg_emb = self.out_embed(negatives)

        pos_score = torch.sum(center_emb * context_emb, dim=1)
        pos_loss = F.logsigmoid(pos_score)

        neg_score = torch.bmm(neg_emb, center_emb.unsqueeze(2)).squeeze()
        neg_loss = F.logsigmoid(-neg_score).sum(1)

        loss = -(pos_loss + neg_loss).mean()
        return loss

    def training_step(self, batch, batch_idx):
        center, context, negatives = batch
        loss = self(center, context, negatives)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        center, context, negatives = batch
        loss = self(center, context, negatives)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
