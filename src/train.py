import torch 
from tqdm.auto import tqdm

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pickle
import os 

class Trainer():

    def __init__(self,
                 model, 
                 loss_fn, optim,
                 device,
                 early_stopper=None, lr_scheduler=None):
        
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim
        self.device = device

        self.early_stopper = early_stopper
        self.lr_scheduler = lr_scheduler

        self.history = {
            "train_losses": [],
            "val_losses": [],

            "train_accs": [],
            "val_accs": [],

            "train_f1s": [],
            "val_f1s": [],

            "train_lrs": []
        }

    def save(self,filepath):
        """
        Saves Trainer History into a .pkl file
        """

        os.mkdirs(filepath, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(self.history, f)


    def _batch_to_device(self, batch):
        
        X,y = batch
        return X.to(self.device), y.to(self.device)

    def _train_batch(self,batch):

        X, y = self._batch_to_device(batch)

        self.optim.zero_grad()

        y_pred_logits = self.model(X)

        loss = self.loss_fn(y_pred_logits, y)
        loss.backward()
        self.optim.step()

        return {
            "loss": loss.item(),
            "pred_logits": y_pred_logits.detach().cpu(), 
            "targets": y.detach().cpu()
        }
    
    def _run_epoch(self,loader, mode = "train"):
        
        modes = ["train", "val", "test"]

        if mode.lower() not in modes:
            raise ValueError("Invalid Mode Passed")
        
        is_training = (mode.lower() == "train")

        losses = []
        pred_logits = []
        targets = []

        for batch in tqdm(loader, leave=False, desc=f"{mode.capitalize()}ing"):

            batch_data = self._train_batch(batch) if is_training else self._evaluate_batch(batch)
            
            losses.append(batch_data["loss"])
            pred_logits.append(batch_data["pred_logits"])
            targets.append(batch_data["targets"])

        # Calculate Avg Loss
        avg_loss = sum(losses) / len(losses)

        # Combine Pred Logits accross the Epoch 
        pred_logits = torch.cat(pred_logits)
        
        # Combine True Labels across the Epoch
        targets = torch.cat(targets) 
            
        return {
            "loss": avg_loss,
            "pred_logits": pred_logits,
            "targets": targets
        }
    
    def _evaluate_batch(self, batch):

        X, y = self._batch_to_device(batch)

        y_pred_logits = self.model(X)
        loss = self.loss_fn(y_pred_logits, y)

        return {
            "loss": loss.item(),
            "pred_logits": y_pred_logits.detach().cpu(), 
            "targets": y.detach().cpu()
        }

    def _calc_and_store_metrics(self, data, mode="train"):

        if mode.lower() not in ["train", "val"]:
            raise ValueError("Mode must be 'train' or 'val'")
        
        preds = torch.argmax(data["pred_logits"], dim=1).numpy()
        targets = data["targets"].numpy()

        acc = accuracy_score(targets, preds)
        f1 = f1_score(targets, preds, average="macro")

        self.history[f"{mode}_losses"].append(data["loss"])
        self.history[f"{mode}_accs"].append(acc)
        self.history[f"{mode}_f1s"].append(f1)

    def train_val_model(self,epochs,
                        train_loader, val_loader,
                        callback = None):      
    
        for epoch in range(epochs):

            self.cur_epoch = epoch + 1
            print(f"Epoch {self.cur_epoch} of {epochs}")

            # Train the Model
            self.model.train() 
            train_epoch_data = self._run_epoch(train_loader, mode="train")

            # Validate the Model
            self.model.eval()
            with torch.inference_mode():
                val_epoch_data = self._run_epoch(val_loader, mode="val")

            # Calculate and Store Metrics 
            self._calc_and_store_metrics(train_epoch_data, "train")
            self._calc_and_store_metrics(val_epoch_data, "val")

            cur_lr = self.optim.param_groups[0]['lr']
            self.history["train_lrs"].append(cur_lr)

            # Print Metrics 
            # Last Element of History corresponds to Current Epochs Metrics 
            print(f"Summary:")
            print(f"Current LR: {cur_lr: .5f}")
            print(f"Train Loss: {self.history['train_losses'][-1]:.3f} | Acc: {self.history['train_accs'][-1]*100:.2f}% | F1-Macro: {self.history['train_f1s'][-1]*100:.2f}%")
            print(f"Val Loss:   {self.history['val_losses'][-1]:.3f}   | Acc:  {self.history['val_accs'][-1]*100:.2f}%  | F1-Macro: {self.history['val_f1s'][-1]*100:.2f}%")
            

            val_metric_to_monitor = self.history['val_losses'][-1]
            
            # This is where we callback
            if callback:
                callback(self)

            # Call Learning Rate Scheduler if it has been implemented 
            if self.lr_scheduler:

                # Use step(metric) for ReduceLROnPlateau, step() for others
                if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.lr_scheduler.step(val_metric_to_monitor)
                else:
                    self.lr_scheduler.step()
 
            # Call Early Stopping if it has been implemented
            if self.early_stopper:
                early_stop = self.early_stopper(model=self.model,
                                                metric=val_metric_to_monitor,
                                                epoch=self.cur_epoch,
                                                optimizer_state_dict=self.optim.state_dict(),
                                                lr_scheduler_state_dict=self.lr_scheduler.state_dict() if self.lr_scheduler else None)
                
                if self.early_stopper.counter == 0 and not early_stop:
                    self.history["best_epoch"] = self.cur_epoch

                if early_stop: 
                    print("Early stop signal received. Exiting training loop...")
                    print(f"{'-'*15}\n")
                    break

            print(f"{'-'*15}\n")
            

    def test_model(self, loader, class_names=None):

        # Validate the Model
        self.model.eval()
        with torch.inference_mode():
            data = self._run_epoch(loader, mode="test")
        
        # Calculate and Print Metrics
        pred_logits = data['pred_logits']
        targets = data['targets']

        predictions = torch.argmax(pred_logits, dim=1).numpy()
        targets = targets.numpy()

        print("\n--- Test Set Classification Report ---")
        report = classification_report(targets, predictions, target_names=class_names)
        print(report)
        print("--------------------------------------\n")

        return data
        


class EarlyStopper():

    def __init__(self, patience, delta, 
                 save_checkpoints=True, checkpoint_path="best_model.pt", 
                 mode="min", # We have a mode so that our EarlyStopper can be metric agnostic 
                 verbose=True): 

        self.checkpoint_path = checkpoint_path  
        self.save_checkpoints = save_checkpoints  
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.early_stop = False

        if mode == "min":
            self.best_score = np.inf
            self.is_better = lambda metric, best_score : metric < (best_score - self.delta)

        elif mode == "max":
            self.best_score = -np.inf
            self.is_better = lambda metric, best_score : metric > (best_score + self.delta)

        else:
            raise ValueError("Mode must be either 'min' or 'max'")
        
        self._print_message(f"Early Stopping initialized with Mode: {self.mode.upper()}")


    def _print_message(self, message):
        if self.verbose:
            print(message)
    

    def __call__(self, model, metric, **others):

        # If there is a sign of improvement
        if self.is_better(metric, self.best_score):

            if(np.isinf(self.best_score)):
                self._print_message(f"Metric: {metric:.4f}")
            else:
                self._print_message(f"Metric improved ({self.best_score:.4f} -> {metric:.4f})")

            if self.save_checkpoints:
                self._print_message(f"Saving best model...")

            self.counter = 0                        # Reset the counter to 0 
            self.best_score = metric                # Set the metric as the new best_score
            if self.save_checkpoints:
                self.save_checkpoint(model, others) # save the current state of the model 


        else:
            self.counter+=1                         # Increase the count
            self._print_message(f"Metric did not improve significantly. Early Stopping Counter: {self.counter} / {self.patience}.")
        
            if self.counter >= self.patience:       # Stop if patience runs out 
                self.early_stop = True

        return self.early_stop
            

    def save_checkpoint(self, model, **others):    # **others is for optimizer_state, etc 
        
        os.mkdirs(self.checkpoint_path, exist_ok=True)

        torch.save({
            "model_state_dict": model.state_dict(),
            **others                                
        }, self.checkpoint_path)