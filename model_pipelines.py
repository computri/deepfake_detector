import torch
import torch.nn as nn
from collections import defaultdict
import pytorch_lightning as pl
import torchmetrics



class DeepFakeClassifier(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-3, report_per_class_accuracy=False):
        super().__init__()
        self.save_hyperparameters()
        self.val_stats = defaultdict(lambda: {"real": {"correct": 0, "total": 0},
                                              "fake": {"correct": 0, "total": 0}})
        
        self.test_stats = defaultdict(lambda: {"real": {"correct": 0, "total": 0},
                                               "fake": {"correct": 0, "total": 0}})

        self.model = model
        self.loss_fn = nn.BCEWithLogitsLoss()

        

    def forward(self, x):
        return self.model(x)
        

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        logits = self(x)

        y = y.float().unsqueeze(1)
        loss = self.loss_fn(logits, y)
        preds = (torch.sigmoid(logits) > 0.5).int()
        acc = (preds == y.int()).float().mean()
    

        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        logits = self(x)

        

        preds = (torch.sigmoid(logits) > 0.5).int()
        acc = (preds == y.int()).float().mean()
    
       
        if self.hparams.report_per_class_accuracy:
            obj_cls = batch[2]
        else:
            obj_cls = torch.zeros_like(y, dtype=torch.int)

        for pred, label, cls_id in zip(preds.squeeze(1).cpu(), y.cpu(), obj_cls.cpu()):
            label_str = "real" if label == 0 else "fake"
            self.val_stats[cls_id.item()][label_str]["total"] += 1
            if pred.item() == label.item():
                self.val_stats[cls_id.item()][label_str]["correct"] += 1
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, on_step=False)

        return acc
      
    def on_validation_epoch_end(self):
        real_correct, real_total = 0, 0
        fake_correct, fake_total = 0, 0

        print("\nPer-class accuracy (val):")
        for cls_id, stats in sorted(self.val_stats.items()):
            real = stats["real"]
            fake = stats["fake"]

            real_acc = real["correct"] / real["total"] if real["total"] > 0 else 0.0
            fake_acc = fake["correct"] / fake["total"] if fake["total"] > 0 else 0.0

            real_correct += real["correct"]
            real_total += real["total"]
            fake_correct += fake["correct"]
            fake_total += fake["total"]

            print(f"Class {cls_id}: REAL acc = {real_acc:.3f}, FAKE acc = {fake_acc:.3f}")

        overall_real_acc = real_correct / real_total if real_total > 0 else 0.0
        overall_fake_acc = fake_correct / fake_total if fake_total > 0 else 0.0
        print(f"\nOverall REAL accuracy: {overall_real_acc:.3f}")
        print(f"Overall FAKE accuracy: {overall_fake_acc:.3f}")
    
    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        logits = self(x)

        preds = (torch.sigmoid(logits) > 0.5).int()
        acc = (preds == y.int()).float().mean()
    
    
        if self.hparams.report_per_class_accuracy:
            obj_cls = batch[2]
        else:
            obj_cls = torch.zeros_like(y, dtype=torch.int)
            
        for pred, label, cls_id in zip(preds.squeeze(1).cpu(), y.cpu(), obj_cls.cpu()):
            label_str = "real" if label == 0 else "fake"
            self.test_stats[cls_id.item()][label_str]["total"] += 1
            if pred.item() == label.item():
                self.test_stats[cls_id.item()][label_str]["correct"] += 1
        self.log("test_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        return acc
    
    def on_test_epoch_end(self):
        real_correct, real_total = 0, 0
        fake_correct, fake_total = 0, 0

        print("\nPer-class accuracy (test):")
        for cls_id, stats in sorted(self.test_stats.items()):
            real = stats["real"]
            fake = stats["fake"]

            real_acc = real["correct"] / real["total"] if real["total"] > 0 else 0.0
            fake_acc = fake["correct"] / fake["total"] if fake["total"] > 0 else 0.0

            real_correct += real["correct"]
            real_total += real["total"]
            fake_correct += fake["correct"]
            fake_total += fake["total"]

            print(f"Class {cls_id}: REAL acc = {real_acc:.3f}, FAKE acc = {fake_acc:.3f}")

        overall_real_acc = real_correct / real_total if real_total > 0 else 0.0
        overall_fake_acc = fake_correct / fake_total if fake_total > 0 else 0.0
        print(f"\nOverall REAL accuracy: {overall_real_acc:.3f}")
        print(f"Overall FAKE accuracy: {overall_fake_acc:.3f}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)