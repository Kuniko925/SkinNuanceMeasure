import torch
import torch.nn as nn
import torch.quantization
from sklearn.metrics import classification_report, f1_score, accuracy_score
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelTrainer():
    def _run_epoch(self, model, loader, criterion, optimizer=None, scheduler=None, batch_scheduler=False, isTrain=False):

        running_loss, all_preds, all_labels = 0.0, [], []
        for batch in loader:
            inputs, labels = batch[0].to(device), batch[1].to(device)

            if isTrain:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if isTrain:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if batch_scheduler and scheduler is not None: scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        avg_loss = running_loss / len(loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average="macro")
        return all_preds, all_labels, acc, f1, avg_loss

    def train(self, model, loaders, training_settings):

        train_loader = loaders['train']
        valid_loader = loaders['valid']

        optimizer = training_settings['optimizer']
        scheduler = training_settings['scheduler']
        epochs = training_settings['epochs']
        model_save_directory = training_settings['model_save_directory']
        batch_scheduler= training_settings['batch_scheduler']

        model.to(device)
        criterion = nn.CrossEntropyLoss()

        best_val_file, best_f1 = None, None
        for epoch in range(epochs):
            # train
            model.train()
            train_pred, _, train_acc, train_f1, train_loss = self._run_epoch(model, train_loader, criterion, optimizer, scheduler, batch_scheduler, isTrain=True)

            # Validation
            model.eval()
            val_pred, _, val_acc, val_f1, val_loss = self._run_epoch(model, valid_loader, criterion, isTrain=False)

            print(
                f"Epoch: {epoch} | "
                f"Train Acc {train_acc:.4f} | F1: {train_f1:.4f} | Loss {train_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | Loss: {val_loss:.4f} | F1: {val_f1:.4f}"
            )

            if best_f1 is None or best_f1 < val_f1 or epoch == epochs - 1:
                if epoch != epochs - 1:
                    best_f1 = val_f1
                    best_val_file = Path(model_save_directory) / f"model_{epoch}.pt"
                torch.save(model.state_dict(), Path(model_save_directory) / f"model_{epoch}.pt")

            if scheduler is not None and not batch_scheduler:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_f1)

        print(best_val_file)
        return best_val_file

    def test(self, model, test_loader):
        model.to(device)
        model.eval()

        criterion = nn.CrossEntropyLoss()
        pred, labels, _, _, _ = self._run_epoch(model, test_loader, criterion, isTrain=False)
        rep = classification_report(labels, pred, output_dict=True)
        print(rep)
        return pred