import torch
import torch.nn as nn
import torch.quantization
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
import matplotlib.pyplot as plt
from regularization import DiffEOpp
from bayesian_regularization import BRFairLoss
PROJECT_ROOT = Path(__file__).parent
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_class_weights(train_loader):
    all_labels = []
    all_groups = []

    for batch in train_loader:
        labels = batch[1]
        groups = batch[2]

        if torch.is_tensor(labels):
            labels = labels.detach().cpu().numpy()
        if torch.is_tensor(groups):
            groups = groups.detach().cpu().numpy()

        all_labels.extend(labels)
        all_groups.extend(groups)

    all_labels = np.asarray(all_labels)
    all_groups = np.asarray(all_groups)

    classes = np.unique(all_labels)
    groups_unique = np.unique(all_groups)

    class_w = compute_class_weight(class_weight="balanced", classes=classes, y=all_labels)
    group_w = compute_class_weight(class_weight="balanced", classes=groups_unique, y=all_groups)

    print("classes (encoded):", classes)
    print("class_weights:", class_w)
    print("groups (encoded):", groups_unique)
    print("group_weights:", group_w)

    class_w = torch.tensor(class_w, dtype=torch.float, device=device)
    group_w = torch.tensor(group_w, dtype=torch.float, device=device)

    return class_w, group_w

class ModelTrainer():
    def __init__(self, train_loader, model_name, train_mode="baseline", brfair_regressors=None, beta=0.01):
        self.class_weights, self.group_weights = get_class_weights(train_loader)
        self.train_method = train_mode
        self.beta = beta
        self.model_name = model_name
        self.fairness_strength = self._get_fairness_strength()
        self.brfair_regressors = brfair_regressors

    def _get_fairness_strength(self):
        if self.train_method == "regularization":
            return 0.1
        elif self.train_method == "individual":
            fairness_strength_map = {
                "ConvNext": 0.1,
                "MobileNetV3": 0.1,
                "FastVit": 0.1,
                "MobileVitV2": 0.1,
                "VitTiny": 0.1,
                "DeitTiny": 0.1,
            }
            return fairness_strength_map[self.model_name]
        else:
            return 0.0

    def compute_loss(self, outputs, labels, groups, wd, criterion, fair_criterion=None, brfair_criterion=None):
        loss = criterion(outputs, labels)
        fair_loss = torch.tensor(0.0, device=outputs.device)

        if self.train_method == "reweight":
            group_w = self.group_weights[groups]
            return (loss * group_w).mean(), fair_loss

        if self.train_method == "regularization":
            fair_loss = fair_criterion(outputs, groups, labels)
            fair_loss = self.fairness_strength * fair_loss
            loss = loss + fair_loss
            loss = loss.mean()
            return loss, fair_loss

        if self.train_method == "individual":
            fair_loss = brfair_criterion(loss, wd, labels)
            loss = fair_loss
            return loss, fair_loss

    def _run_epoch(self, model, loader, loss_functions, optimizer=None, scheduler=None, batch_scheduler=False, isTrain=False):

        criterion = loss_functions[0]
        fair_criterion = loss_functions[1]
        brfair_criterion = loss_functions[2]

        running_loss, fair_running_loss, all_preds, all_labels, all_pred_logits, all_label_logits, all_probs = 0.0, 0.0, [], [], [], [], []
        for batch in loader:
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            groups = batch[2].to(device) # skin tone
            wd = batch[3].to(device) # wd

            if isTrain:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss, fair_loss = self.compute_loss(outputs, labels, groups, wd, criterion, fair_criterion, brfair_criterion)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if batch_scheduler and scheduler is not None: scheduler.step()

            else:
                with torch.no_grad():
                    outputs = model(inputs)
                    loss, fair_loss = self.compute_loss(outputs, labels, groups, wd, criterion, fair_criterion, brfair_criterion)

            running_loss += loss.item() * inputs.size(0)
            fair_running_loss += fair_loss.item() * inputs.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())
            all_probs.append(probs.detach().cpu())

            pred_logits = outputs.gather(1, preds.unsqueeze(1)).squeeze(1)
            label_logits = outputs.gather(1, labels.unsqueeze(1)).squeeze(1)
            all_pred_logits.append(pred_logits.detach().cpu())
            all_label_logits.append(label_logits.detach().cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()
        all_probs = torch.cat(all_probs).numpy()
        all_pred_logits = torch.cat(all_pred_logits).numpy()
        all_label_logits = torch.cat(all_label_logits).numpy()

        avg_loss = running_loss / len(loader.dataset)
        avg_fair_loss = fair_running_loss / len(loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        return all_preds, all_labels, all_pred_logits, all_label_logits, all_probs, acc, f1, avg_loss, avg_fair_loss

    def train(self, model, loaders, training_settings):

        train_loader = loaders['train']
        valid_loader = loaders['valid']

        optimizer = training_settings['optimizer']
        scheduler = training_settings['scheduler']
        epochs = training_settings['epochs']
        model_save_directory = training_settings['model_save_directory']
        batch_scheduler= training_settings['batch_scheduler']

        model.to(device)
        reduction = 'mean' if self.train_method == 'baseline' else 'none'
        criterion = nn.CrossEntropyLoss(weight=self.class_weights, reduction=reduction, label_smoothing=0.01)
        fair_criterion = DiffEOpp()
        brfair_criterion = BRFairLoss(self.brfair_regressors)
        loss_functions = [criterion, fair_criterion, brfair_criterion]

        history = HistoryLogger()
        best_val_file, best_f1 = None, None

        for epoch in range(epochs):
            # train
            model.train()
            train_pred, _, _, _, _, train_acc, train_f1, train_loss, train_fair_loss = self._run_epoch(model, train_loader, loss_functions, optimizer, scheduler, batch_scheduler, isTrain=True)

            # Validation
            model.eval()
            val_pred, _, _, _, _, val_acc, val_f1, val_loss, val_fair_loss = self._run_epoch(model, valid_loader, loss_functions, isTrain=False)

            lr = optimizer.param_groups[0]["lr"]
            history.log(epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1, lr)

            print(
                f"Epoch: {epoch} | "
                f"Train Acc {train_acc:.4f} | F1: {train_f1:.4f} | Loss {train_loss:.4f} | Fair Loss {train_fair_loss:.4f} |"
                f"Val Acc: {val_acc:.4f} | Loss: {val_loss:.4f} | F1: {val_f1:.4f} | lr={lr:.2e} |"
            )

            if best_f1 is None or best_f1 < val_f1 or epoch == epochs - 1:
                if epoch != epochs - 1:
                    best_f1 = val_f1
                    best_val_file = Path(model_save_directory) / f"model_{epoch}.pt"
                torch.save(model.state_dict(), Path(model_save_directory) / f"model_{epoch}.pt")

            if scheduler is not None and not batch_scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_f1)
                else:
                    scheduler.step()

        history.plot(f'{Path(model_save_directory)} training_process.png')
        print(best_val_file)
        return best_val_file

    def test(self, model, test_loader):
        model.to(device)
        model.eval()

        criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        fair_criterion = DiffEOpp()
        brfair_criterion = BRFairLoss(self.brfair_regressors)
        loss_functions = [criterion, fair_criterion, brfair_criterion]

        pred, labels, pred_logits, label_logits, prob, _, _, _, _ = self._run_epoch(model, test_loader, loss_functions, self.beta, isTrain=False)
        print(classification_report(labels, pred))
        return pred, labels, pred_logits, label_logits, prob

@dataclass
class HistoryLogger:
    epochs: list = field(default_factory=list)
    train_loss: list = field(default_factory=list)
    train_acc: list = field(default_factory=list)
    train_f1: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_acc: list = field(default_factory=list)
    val_f1: list = field(default_factory=list)
    lr: list = field(default_factory=list)

    def log(self, epoch, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1, lr):
        self.epochs.append(epoch)
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.train_f1.append(train_f1)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)
        self.val_f1.append(val_f1)
        self.lr.append(lr)

    def plot(self, filename="training_curves.png"):
        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # Loss (train vs val)
        axes[0].plot(self.epochs, self.train_loss, label="train")
        axes[0].plot(self.epochs, self.val_loss, label="val")
        axes[0].set_ylabel("loss")
        axes[0].legend()
        axes[0].grid(True)

        # Accuracy (train vs val)
        axes[1].plot(self.epochs, self.train_acc, label="train")
        axes[1].plot(self.epochs, self.val_acc, label="val")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_xlabel("epoch")
        axes[1].legend()
        axes[1].grid(True)

        axes[2].plot(self.epochs, self.train_f1, label="train")
        axes[2].plot(self.epochs, self.val_f1, label="val")
        axes[2].set_ylabel("F1")
        axes[2].set_xlabel("epoch")
        axes[2].legend()
        axes[2].grid(True)

        fig.tight_layout()
        fig.savefig(filename, dpi=200, bbox_inches="tight")
        plt.close(fig)