import torch
import torch.nn.functional as F

class Evaluate (torch.nn.Module):
    def __init__ (self, dataset=None, task='classification', metric_name='ACC'):
        super (Evaluate, self).__init__()
        self.dataset = dataset
        self.task = task 
        self.metric_name = metric_name

    def predict_label (self, pred):
        if 'bin' in self.task or 'link' in self.task:
            return (pred > 0.5).int()
        else: 
            return pred.argmax(dim=-1)

    def forward (self, y_hat, y):
        # y_hat is logits
        if self.metric_name == 'ACC':
            y_pred = self.predict_label (y_hat)
            return ((y_pred == y).sum()/y.shape[0])
        elif self.metric_name == 'report':
            from sklearn.metrics import classification_report
            y_pred = self.predict_label (y_hat)
            return (classification_report (y.cpu(), y_pred.cpu()))
        elif self.metric_name == 'f1_macro':
            from sklearn.metrics import f1_score
            y_pred = self.predict_label (y_hat)
            return (f1_score (y.cpu(), y_pred.cpu(), average='macro'))
        elif self.metric_name == 'f1_micro':
            from sklearn.metrics import f1_score
            y_pred = self.predict_label (y_hat)
            return (f1_score (y.cpu(), y_pred.cpu(), average='micro'))
        elif self.metric_name == 'f1_wtd':
            from sklearn.metrics import f1_score
            y_pred = self.predict_label (y_hat)
            return (f1_score (y.cpu(), y_pred.cpu(), average='weighted'))
        elif self.metric_name == 'AUC':
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y.cpu(), y_hat.cpu())
            return auc
        