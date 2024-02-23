import numpy as np
import torch
import sklearn.metrics as sk_metrics


class MetricsAll:

    def __init__(self, class_num, name='MetricsAll', axis=-1):
        super(MetricsAll, self).__init__()
        self.axis = axis
        self.name = name
        self.class_num = class_num
        self.labels = []

    def reset(self):
        self.labels = []
        return

    def update(self, labels, preds):
        if isinstance(labels, torch.Tensor):
            labels = [labels]
        if isinstance(preds, torch.Tensor):
            preds = [preds]

        for label, pred_label in zip(labels, preds):
            if pred_label.shape != label.shape:
                logits = torch.softmax(pred_label,dim=self.axis)
                pred_score,_ = torch.max(logits, axis=self.axis)
                pred_label = torch.argmax(pred_label, axis=self.axis)
            pred_label = pred_label.numpy().astype('int32')
            label = label.numpy().astype('int32')
            score = pred_score.numpy()
            
            label = label.flat
            score = score.flat
            pred_label = pred_label.flat

            for a, b,s in zip(label, pred_label,score):
                a, b = int(a), int(b)
                self.labels.append((a,b,s))
    def get(self,name="f1score"):
        labels = [k for k in range(self.class_num)]
        if name.lower() == "f1score":
            y_true = [a[0] for a in self.labels]
            y_pred = [a[1] for a in self.labels]
            return name.lower(), sk_metrics.f1_score(y_true, y_pred,labels= labels, average='macro')
        if name.lower() == "accuracy":
            y_true = [a[0] for a in self.labels]
            y_pred = [a[1] for a in self.labels]
            return name.lower(),sk_metrics.accuracy_score(y_true, y_pred) 
        if name.lower() == "recalling":
            y_true = [a[0] for a in self.labels]
            y_pred = [a[1] for a in self.labels]
            return name.lower(),sk_metrics.recall_score(y_true, y_pred,average="macro") 
        if name.lower() == "precision":
            y_true = [a[0] for a in self.labels]
            y_pred = [a[1] for a in self.labels]
            return name.lower(),sk_metrics.precision_score(y_true, y_pred,average="macro") 
        if name.lower() == "classification_report":
            y_true = [a[0] for a in self.labels]
            y_pred = [a[1] for a in self.labels]
            return name.lower(), sk_metrics.classification_report(y_true, y_pred,labels=labels) # sk_metrics.accuracy_score(y_true, y_pred)
        if name.lower() == "confusion_matrix":
            y_true = [a[0] for a in self.labels]
            y_pred = [a[1] for a in self.labels]
            return name.lower(), sk_metrics.confusion_matrix(y_true,y_pred,labels=labels,normalize=None) 
        if name.lower() == "classification_report_dict":
            y_true = [a[0] for a in self.labels]
            y_pred = [a[1] for a in self.labels]
            return name.lower(), sk_metrics.classification_report(y_true,y_pred,labels=labels,output_dict=True) 
        return "unk",0.0
     
    def get_visual_metrics_with_confidence(self,thresholds):
        y_true = [a[0] for a in self.labels]
        labels = [-1] + [k for k in range(self.class_num)]
        num_classes = self.class_num + 1
        rets = []
        for th in thresholds:
            y_pred = [b if s > th else -1 for (_,b,s) in self.labels]
            cm = sk_metrics.confusion_matrix(y_true, y_pred,normalize=None,labels=labels)
            
            num = cm.sum()
            tp_plus_tn = np.diag(cm).sum()
            accuracy = tp_plus_tn / (num if num > 0 else 1e-9)
            
            precisions, recalls = [], []
            for c in range(1, num_classes):
                tp = cm[c,c]
                tp_plus_fp = cm[:,c].sum()
                tp_plus_fn = cm[c,:].sum()
                p = tp / tp_plus_fp if tp_plus_fp > 0 else 1e-9
                r = tp / tp_plus_fn if tp_plus_fn > 0 else 1e-9
                precisions.append(p)
                recalls.append(r)
            precision = np.mean(precisions)
            recall = np.mean(recalls)
            rets.append(
                {"cm":cm,"recall":recall,'accuracy':accuracy, "precision":precision}
            )
        return rets
                 
