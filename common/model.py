import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from .callbacks import LRScheduler, EarlyStopping, StoreBestModel
   
class Model(nn.Module):
    """
    A neural network model for training and evaluation, inheriting from PyTorch's nn.Module.

    Attributes:
        activation_layers (list): List to hold activation layers.
        optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        loss (callable): The loss function used for training.
        device (torch.device): The device (CPU or GPU) on which the model is being trained.
    """
    
    def __init__(self, predefined_model=None):
        """
        Initializes the Model class, setting up the basic attributes.
        """
        super(Model, self).__init__()
        self.activation_layers = []
        self.optimizer = None
        self.loss = None
        self.clipping = False
        self.layers = None
        self.callbacks = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if predefined_model is not None:
            self.layers = predefined_model
            self.forward = predefined_model.forward
   
    def compile(self, loss, optimizer, optimizer_params={}, metrics=None):
        """
        Configures the model for training by specifying the loss function, optimizer, and metrics.
        
        Args:
            loss (str or callable): Loss function name or callable (e.g., 'categorical_crossentropy' or nn.CrossEntropyLoss).
            optimizer (str, callable, or tuple): Optimizer name, callable, or a tuple with the optimizer and its parameters.
            metrics (list of str, optional): List of metrics to track during training.
        """
        if isinstance(loss, str):
            if loss == "categorical_crossentropy":
                self.loss = nn.CrossEntropyLoss()
            else:
                raise ValueError(f"Loss '{loss}' not recognized.")
        elif isinstance(loss, nn.Module):
            self.loss = loss
        else:
            raise ValueError("Loss must be a string or an instance of nn.Module.")

        # Set the optimizer
        if isinstance(optimizer, str):
            if optimizer.lower() == "adam":
                self.optimizer = optim.Adam(self.parameters(), **optimizer_params)  
            elif optimizer.lower() == "sgd":
                self.optimizer = optim.SGD(self.parameters(), **optimizer_params)   
            else:
                raise ValueError(f"Optimizer '{optimizer}' not recognized.")
        elif isinstance(optimizer, torch.optim.Optimizer):
            self.optimizer = optimizer 
        elif isinstance(optimizer, type):
            self.optimizer = optimizer(self.parameters(), **optimizer_params) 
        else:
            raise ValueError("Optimizer must be a string or an instance of torch.optim.Optimizer.")
        self.metrics = metrics
        
    def add_callback(self, callbacks):
        """
        Add callbacks to the model's training process.
        Parameters:
        callbacks (list or dict): A list of callback configurations or a single callback configuration as a dictionary.
            Each callback configuration should contain a "type" key specifying the type of callback and other
            necessary parameters for that callback type.
        Supported callback types:
        - "early_stopping": Adds an EarlyStopping callback.
            - patience (int, optional): Number of epochs with no improvement after which training will be stopped. Default is 10.
            - delta (float, optional): Minimum change in the monitored quantity to qualify as an improvement. Default is 0.
        - "storage": Adds a StoreBestModel callback.
            - path (str): Path where the best model will be stored.
        - "scheduler": Adds an LRScheduler callback.
            - scheduler_class (str): The class name of the learning rate scheduler from torch.optim.lr_scheduler.
            - scheduler_params (dict, optional): Parameters to initialize the scheduler.
            - lr_metric (str, optional): Metric to monitor for learning rate adjustments. Default is "val_loss".
        """
        if isinstance(callbacks, dict):  
            callbacks = [callbacks]
            
        for callback_config in callbacks:
            callback_type = callback_config.get("type")
            
            if callback_type == "early_stopping":
                if not any(isinstance(cb, EarlyStopping) for cb in self.callbacks):
                    patience = callback_config.get("patience", 10)
                    delta = callback_config.get("delta", 0)
                    self.callbacks.append(EarlyStopping(patience=patience, delta=delta))
                    
            elif callback_type == "storage":
                if not any(isinstance(cb, StoreBestModel) for cb in self.callbacks):
                    metric = self.metrics
                    path = callback_config.get("path")
                    self.callbacks.append(StoreBestModel(metric, path))
                    
            elif callback_type == "scheduler":
                if not any(isinstance(cb, LRScheduler) for cb in self.callbacks):
                    scheduler_class = callback_config.get("scheduler_class")   
                    scheduler_params = callback_config.get("scheduler_params", {})   
                    self.lr_metric = callback_config.get("lr_metric", "val_loss")
                    
                if not scheduler_class:
                    raise ValueError("You must specify a 'scheduler_class' to use a scheduler.")

                scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_class, None)

                if scheduler_cls is None:
                    raise ValueError(f"Scheduler '{scheduler_class}' not found in torch.optim.lr_scheduler.")

                scheduler = scheduler_cls(optimizer=self.optimizer, **scheduler_params)
                self.callbacks.append(LRScheduler(scheduler))
            else:
                raise ValueError(f"Callback type '{callback_type}' unknown.")

    def summary(self):
        """
        Prints a summary of the model architecture, loss function, optimizer, and metrics.
        """
        print("Model Summary:")
        print(self)
        
        if self.loss is not None:
            print(f"Loss Function: {self.loss}")
        
        if self.optimizer is not None:
            print(f"Optimizer: {self.optimizer}")
        
        if hasattr(self, 'metrics'):
            print(f"Metrics: {self.metrics}")
        
        if hasattr(self, 'total_paths'):
            print(f"Total Paths: {self.total_paths}")
            
    def train_model(self, train_data,  val_data=None, num_epochs=10, verbose=True):        
        """
        Trains the model on the training data for a specified number of epochs.

        Args:
            train_data (DataLoader): The training data loader providing batches of input and targets.
            val_data (DataLoader, optional): The validation data loader for evaluating the model during training. Default is None.
            num_epochs (int): The number of epochs for which to train the model. Default is 10.
        """
        device = self.device
        self.to(device)
        history = {
            "train_metric": [],   
            "train_loss": [],
            "val_metric": [],   
            "val_loss": []
        }
        for epoch in range(num_epochs):  
            actual_lr = self.optimizer.param_groups[0]['lr']
            # Train
            if epoch == 0:
                self.train_loss = self.training_one_epoch(train_data,num_epochs, epoch, verbose=True)
            else:
                self.train_loss = self.training_one_epoch(train_data,num_epochs, epoch, verbose=False)
            # Evaluate
            val_info = self.evaluate_model(val_data, verbose=False)
            self.val_loss = val_info["loss"]
            
            # Callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(self)
            
            if verbose:
                print(f'Epoch {epoch + 1:3d}/{num_epochs}: {self.metrics} - learning rate: {actual_lr:.4f}', flush=True)
            
            history["train_metric"].append(self.metrics.train_acc)   
            history["train_loss"].append(self.train_loss)
            history["val_metric"].append(self.metrics.val)
            history["val_loss"].append(self.val_loss)
            
            # In case of EarlyStopping
            if hasattr(self, "stop_training") and self.stop_training:
                self.stop_training = False
                break
        
        for callback in self.callbacks:
                callback.on_train_end(self)
        self.metrics.reset()
        return history

    def training_one_epoch(self, train_data, num_epochs, epoch, verbose):
        """
        Runs a single training epoch over the training data.

        Args:
            train_data (DataLoader): The training data loader.
            device (torch.device): The device to which the model should be moved (CPU or GPU).
            epoch (int): The current epoch number.
            num_epochs (int): The total number of epochs for training.
        """
        self.train()
        running_loss = 0.0
        total_loss = 0.0
        self.metrics.reset_train()
        
        n_steps = len(train_data)
        for i, batch in enumerate(train_data):
            outputs, loss = self.train_step(batch)
            targets = batch[1].to(self.device)

            # Update loss and metrics
            total_loss += loss  
            running_loss += loss  
            self.metrics.update_train(outputs, targets)
            
            if ((i + 1) % 20 == 0) and verbose:
                avg_loss = running_loss / 20
                print(f'Epoch {epoch + 1:3d}/{num_epochs} - Step {i+1}/{n_steps}: Train Acc: {self.metrics.train_acc:.2f}%, Loss: {avg_loss:.4f}', end='\n', flush=True)
                running_loss = 0
        
        return total_loss / len(train_data)
    
    def evaluate_model(self, val_data, verbose=True):
        """
        Evaluates the model on the validation data.

        Args:
            val_data (DataLoader): The validation data loader.
            verbose (bool, optional): If True, prints the validation accuracy. Defaults to False.
        Returns:
            float: The validation accuracy if verbose is False.
        """
   
   
        self.to(self.device)
        self.eval()   
        val_loss = 0.0
        self.metrics.reset_val()
        if val_data:
            with torch.no_grad():  
                for batch in val_data:
                    outputs, loss = self.evaluate_step(batch)
                    targets = batch[1].to(self.device)

                    val_loss += loss
                    self.metrics.update_val(outputs, targets)

            if verbose:
                print(self.metrics)
                
            return {"loss": val_loss / len(val_data), "metric": self.metrics.val}
            
    def train_step(self, batch):
        inputs, targets = batch
        inputs, targets = inputs.to(self.device), targets.to(self.device) 
        self.optimizer.zero_grad()
        outputs = self.forward(inputs)
        loss = self.loss(outputs, targets)
        loss.backward()
        if self.clipping: nn.utils.clip_grad_norm_(self.parameters(), 1)
        self.optimizer.step()
        return outputs, loss.item()
    
    def evaluate_step(self, batch):
        with torch.no_grad():  # Disable gradient calculation for testing
            inputs, targets = batch
            inputs, targets = inputs.to(self.device), targets.to(self.device) 
            outputs = self.forward(inputs)
            loss = self.loss(outputs, targets)
        return outputs, loss.item()
                      
    def predict(self, data, batch_size=1):
        """
        Makes predictions on input tensor, batch of tensors, or DataLoader.

        Args:
            data (torch.Tensor or DataLoader): A single input tensor, batch of tensors, or DataLoader.
            batch_size (int): Batch size for inference if data is a tensor.

        Returns:
            list: A list of predicted targets.
        """
        device = self.device
        self.to(device)
        self.eval()
        predictions = []
        all_outputs = []
        with torch.no_grad():
            if isinstance(data, DataLoader):
                dataloader = data
            else:
                if isinstance(data, torch.Tensor):
                    if data.dim() == 3:  
                        data = data.unsqueeze(0)   
                    dataset = TensorDataset(data)
                    dataloader = DataLoader(dataset, batch_size=batch_size)
                else:
                    print("The argument data must be a tensor or a DataLoader.")
                    return []

            # Inference loop
            for batch in dataloader:
                inputs = batch[0].to(device)   
                outputs = self.forward(inputs)  
                _ , predicted = torch.max(outputs, 1)   
                
                predictions.extend(predicted.cpu().numpy())   
                all_outputs.append(outputs.cpu())  
                
        return predictions, torch.cat(all_outputs, dim=0)
    
    def save_model(self, path='model.pth', verbose=True):
        """
        Saves the model state and optimizer state to a file.

        Args:
            path (str, optional): The file path where the model will be saved. Defaults to 'model.pth'.
            verbose (bool, optional): If True, prints a message confirming the model has been saved. Defaults to True.
        """
        state = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'trainable_params': {name: param.requires_grad for name, param in self.named_parameters()}
        }
        torch.save(state, path)
        if verbose:
            print(f"Model saved to {path}")

    def load_model(self, path='model.pth', verbose=False):
        """
        Loads the model and optimizer state from a checkpoint file.
        
        Args:
            path (str): The path to the checkpoint file. Default is 'model.pth'.
            verbose (bool): If True, prints a message indicating the model has been loaded and the starting epoch. Default is False.
        """
        device = self.device
        checkpoint = torch.load(path, map_location=torch.device(device), weights_only=False)
        self.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'trainable_params' in checkpoint:
            for name, param in self.named_parameters():
                if name in checkpoint['trainable_params']:
                    param.requires_grad = checkpoint['trainable_params'][name]
        if verbose:
            print(f"Model loaded from {path}")
            
            
    