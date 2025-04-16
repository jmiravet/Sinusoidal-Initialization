import torch
from abc import ABC, abstractmethod

class Metric(ABC):
    @abstractmethod
    def reset(self):
        pass
    @abstractmethod
    def reset_train(self):
        pass
    @abstractmethod
    def reset_val(self):
        pass
    @abstractmethod
    def update_train(self, output, expected):
        pass
    @abstractmethod
    def update_val(self, output, expected):
        pass
    @abstractmethod
    def __str__(self):
        pass

class Metrics(Metric):
    def __init__(self, metrics_list):
        self.metrics_list = metrics_list

    def reset(self):
        for metric in self.metrics_list:
            metric.reset()

    def reset_train(self):
        for metric in self.metrics_list:
            metric.reset_train()

    def reset_val(self):
        for metric in self.metrics_list:
            metric.reset_val()

    def update_train(self, output, expected):
        best = []
        for metric in self.metrics_list:
            best.append(metric.update_train(output, expected))
        return any(best)
    
    def update_val(self, output, expected):
        best = []
        for metric in self.metrics_list:
            best.append(metric.update_val(output, expected))
        return any(best)
    
    def __str__(self):
        string = ""
        for metric in self.metrics_list:
            string += metric.__str__()
        return string

class Accuracy(Metric):
    def __init__(self):
        self.minimize = False
        self.train_correct = 0
        self.train_total = 0
        self.train_acc = 0
        self.best_train = 0
        self.train_updated = False
        self.val_correct = 0
        self.val_total = 0
        self.val = 0
        self.best_val = 0
        self.val_updated = False
        self.reset()

    def reset(self):
        self.reset_train()
        self.reset_val()

    def reset_train(self):
        if self.best_train < self.train_acc:
            self.best_train = self.train_acc
        self.train_correct = 0
        self.train_total = 0
        self.train_acc = 0
        self.train_updated = False

    def reset_val(self):
        if self.best_val < self.val:
            self.best_val = self.val
        self.val_correct = 0
        self.val_total = 0
        self.val = 0
        self.val_updated = False

    def update_train(self, output, expected):
        self.train_updated = True
        _, predict = torch.max(output, 1)
        self.train_correct += (predict == expected).sum().item()
        self.train_total += expected.size(0)
        self.train_acc = 100 * self.train_correct / self.train_total

    def update_val(self, output, expected):
        self.val_updated = True
        _, predict = torch.max(output, 1)
        self.val_correct += (predict == expected).sum().item()
        self.val_total += expected.size(0)
        self.val = 100 * self.val_correct / self.val_total

    def __str__(self):
        string = ""
        if self.train_updated:
            string += f"Train Acc: {self.train_acc:.2f}%. "
        if self.val_updated:
            string += f"Val Acc: {self.val:.2f}%."
        return string