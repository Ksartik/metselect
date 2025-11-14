import math
import torch
import torch.nn.functional as F

class Loss (torch.nn.Module):
    def __init__ (self, dataset=None, fn_name='ce', alpha=0.5, neg_lambda=1.0, n_impostors=-1, eval_loss='ce', class_weights=None, split_loss=False, only_closest_layer=False, batch_hard_thresh=100):
        super (Loss, self).__init__()
        self.dataset = dataset # can be used to create a weighted loss
        self.fn_name = fn_name
        self.num_epochs = 0 
        if fn_name == 'bce':
            self.fn = torch.nn.BCELoss()
        elif fn_name == 'ce':
            self.fn = torch.nn.NLLLoss()
        elif fn_name == 'magnet':
            self.dataset = dataset
            self.alpha = alpha
            self.n_impostors = n_impostors
            self.neg_lambda = neg_lambda
            self.split_loss = split_loss
            self.only_closest_layer = only_closest_layer
            self.batch_hard_thresh = batch_hard_thresh
            if eval_loss == 'bce':
                self.eval_loss = torch.nn.BCELoss()
            elif eval_loss == 'ce':
                self.eval_loss = torch.nn.NLLLoss() 
        self.class_weights = torch.ones(dataset.num_classes).to(dataset.device) if class_weights is None else class_weights

    def forward (self, y_hat, y, evaluate=False):
        if self.fn_name == 'magnet':
            if evaluate:
                return self.eval_loss(y_hat, y)
            # y_hat is B x L x K and y is B x 1 
            # y_hat[i,j,k] = 1/(2*var) || z_i - mu_{j, k} ||2. (dist)
            B, L, K = y_hat.shape
            if self.n_impostors > 0:
                loss = self.alpha # alpha
                loss += y_hat[torch.arange(y.shape[0]), y, :].min(dim=-1)[0] # closest distance
                dst_kmin, dst_kmin_idx = torch.topk(y_hat.reshape(B, L*K), k=self.k_nearest, largest=False, dim=-1)
                # dst_kmin_labels = torch.div(dst_kmin_idx, K, rounding_mode='floor')
                dst_kmin_labels = torch.floor_divide(dst_kmin_idx, K)
                # nearest impostor not c
                neg_loss = (torch.exp(-dst_kmin) * (dst_kmin_labels != y[:, None]).float()).sum(dim=-1)#/((L-1)*K)
                loss += torch.log(neg_loss)
                loss = torch.clip(loss * self.class_weights[y], min=0).mean(dim=0) #hinge loss
            else:
                if self.split_loss:
                    loss = self.alpha # alpha
                    # loss += - torch.logsumexp (-y_hat[torch.arange(y.shape[0]), y, :], dim=-1) # closest distance
                    loss += y_hat[torch.arange(y.shape[0]), y, :].min(dim=-1)[0].repeat(L-1, 1).T # closest distance
                    notc_mask = torch.ones_like(y_hat[:, :, 0], dtype=bool)
                    notc_mask[torch.arange(y.shape[0]), y] = False
                    loss += self.neg_lambda * (torch.logsumexp(-y_hat[notc_mask].reshape(B, L-1, K), dim=-1))
                    loss = torch.clip(loss, min=0).mean() #hinge loss
                    # weights_Lm1 = self.class_weights.repeat(B, 1)[notc_mask].reshape(B, L-1)
                    # loss = torch.clip(loss * self.class_weights[y, None] * weights_Lm1, min=0).sum()/ (self.class_weights[y, None] * weights_Lm1).sum() #hinge loss
                elif self.only_closest_layer:
                    notc_mask = torch.ones_like(y_hat[:, :, 0], dtype=bool)
                    notc_mask[torch.arange(y.shape[0]), y] = False
                    closest_indices = y_hat[torch.arange(y.shape[0]), y, :].detach().min(dim=-1)[1]
                    pos_loss = y_hat[torch.arange(y.shape[0]), y, closest_indices]
                    neg_loss = self.neg_lambda * (torch.logsumexp(-y_hat[torch.arange(y.shape[0]), :, closest_indices][notc_mask].reshape(B, L-1), dim=-1)) # - math.log(K*(L-1)))
                    loss = (pos_loss + neg_loss + self.alpha).relu().mean(dim=0) #/neg_loss.mean().detach()
                else:
                    notc_mask = torch.ones_like(y_hat, dtype=bool)
                    weights = self.class_weights.repeat(y_hat.shape[0], y_hat.shape[2], 1).transpose(1, 2)
                    notc_mask[torch.arange(y.shape[0]), y, :] = False
                    y_hat_notc_all = y_hat[notc_mask].reshape(B, -1)
                    # pos_loss = torch.logsumexp (y_hat[torch.arange(y.shape[0]), y, :], dim=-1) # closest distance
                    pos_loss = y_hat[torch.arange(y.shape[0]), y, :].min(dim=-1)[0]
                    if self.num_epochs < self.batch_hard_thresh:
                        # batch all
                        neg_loss = self.neg_lambda * (torch.logsumexp(-y_hat_notc_all, dim=-1)) # - math.log(K*(L-1)))
                    else:
                        # batch hard
                        neg_indices = y_hat_notc_all.detach().min(dim=-1)[1]
                        neg_loss = - self.neg_lambda * (y_hat_notc_all[torch.arange(neg_indices.shape[0]), neg_indices])
                        # neg_loss = self.neg_lambda * (torch.logsumexp(-y_hat_notc_all, dim=-1)) # - math.log(K*(L-1)))
                    # neg_loss = self.neg_lambda * torch.log((weights[notc_mask].reshape(B, -1) * torch.exp(-y_hat_notc_all)).sum(dim=-1)) 
                    loss = (pos_loss + neg_loss + self.alpha).relu().mean(dim=0) #/neg_loss.mean().detach()
                    # loss = torch.clip(loss * self.class_weights[y], min=0).mean(dim=0) #hinge loss
                self.num_epochs += 1
            return loss
        return self.fn (y_hat, y)
