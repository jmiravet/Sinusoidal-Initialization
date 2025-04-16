import torch
from abc import ABC
class Callback(ABC):
    """
    Abstract base class for defining callbacks in a training pipeline.

    Callbacks provide hooks that allow users to insert custom behavior at various points 
    during training, evaluation, and batch processing.
    """
    def on_train_start(self, model) -> None:
        pass

    def on_train_end(self, model) -> None:
        pass

    def on_eval_start(self, model) -> None:
        pass

    def on_eval_end(self, model) -> None:
        pass

    def on_batch_start(self, model) -> None:
        pass

    def on_batch_end(self, model) -> None:
        pass

    def on_epoch_start(self, model) -> None:
        pass

    def on_epoch_end(self, model) -> None:
        pass


class StoreBestModel(Callback):
    """
    Callback to store the best model during training based on a specified metric.
    
    Attributes:
    -----------
         metric (Metric): The metric used to evaluate the model's performance.
        path (str): The file path where the best model's state_dict will be saved.
        minimize (bool): Whether to minimize or maximize the metric.
        best_result (float): The best value of the metric observed so far.
    Methods:
    -----------
        on_epoch_end(model):
            Checks the current value of the metric and saves the model's state_dict
            if it is the best observed value so far.
        on_train_end(model):
            Loads the best model's state_dict from the specified file path.
    """
    def __init__(self, metric, path):
        super(StoreBestModel, self).__init__()
        self.metric = metric
        self.minimize = self.metric.minimize
        self.best_result = self.metric.best_val
        self.path = path
  
    def on_epoch_end(self, model):
        if (self.minimize and self.metric.val < self.best_result) or \
           (not self.minimize and self.metric.val > self.best_result):
            self.best_result = self.metric.val
            torch.save(model.state_dict(), self.path)

    def on_train_end(self, model):
        model.load_state_dict(torch.load(self.path, weights_only=True)) 

class LRScheduler(Callback):
    """
    Callback that updates the learning rate using a given scheduler at the end of each epoch.

    Args:    
    -----------
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler to be used.

    Methods:
    -----------
        on_epoch_end(model):
            Updates the learning rate at the end of each epoch.
    """
    def __init__(self, scheduler):
        super(LRScheduler, self).__init__()
        self.scheduler = scheduler

    def on_epoch_end(self, model):
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if model.lr_metric == "val_loss":
                self.scheduler.step(model.val_loss)  
            else:
                self.scheduler.step(model.train_loss)   
        else:
            self.scheduler.step()

class EarlyStopping(Callback):
    """
    Callback to stop training when a monitored metric has stopped improving.

    Parameters:
    -----------
        patience : int, optional (default=10)
            Number of epochs with no improvement after which training will be stopped.
        delta : float, optional (default=0.01)
            Minimum change in the monitored metric to qualify as an improvement.
    Attributes:
    -----------
        patience : int
            Number of epochs with no improvement after which training will be stopped.
        best_loss : float or None
            The best recorded value of the monitored metric.
        iterations_without_improvement : int
            Number of epochs since the last improvement in the monitored metric.
        best_model_wts : dict or None
            The state dictionary of the best model weights.

    Methods:
    --------
        on_epoch_end(model):
            Checks if the monitored metric has improved and updates the state accordingly.
            If there is no improvement for a number of epochs greater than `patience`, stops the training 
            and restores the best model weights.
    """
    def __init__(self, patience=10, delta=0.01):
        super(EarlyStopping, self).__init__()
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.iterations_without_improvement = 0
        self.best_model_wts = None
        
    def on_epoch_end(self, model):
        if self.best_loss is None or (self.best_loss - model.val_loss) > self.delta:
            self.best_loss = model.val_loss
            self.iterations_without_improvement = 0
            self.best_model_wts = model.state_dict()
            model.stop_training = False
        else:
            self.iterations_without_improvement += 1
            if self.iterations_without_improvement > self.patience:
                model.load_state_dict(self.best_model_wts) 
                model.stop_training = True
            

        



