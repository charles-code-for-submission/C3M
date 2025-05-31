
import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    precision_score,
    roc_curve
)

from torch_geometric.data import Data, DataLoader as GeoDataLoader



def get_patient_indices(plist1, plist2, patient_ids): 
    id_to_idx1 = {pid: i for i, pid in enumerate(plist1)}
    id_to_idx2 = {pid: i for i, pid in enumerate(plist2)}

    matched = []
    for pid in patient_ids:
        if pid in id_to_idx1 and pid in id_to_idx2:
            matched.append(pid)

    return id_to_idx1, id_to_idx2, matched  


def compute_triplet_loss(node_emb, triplet_index, margin=3.0):
    anchor_idx = triplet_index[0]
    pos_idx = triplet_index[1]
    neg_idx = triplet_index[2]

    anchor_emb = node_emb[anchor_idx]  # [num_triplets, dim]
    pos_emb = node_emb[pos_idx]
    neg_emb = node_emb[neg_idx]

    pos_dist = F.pairwise_distance(anchor_emb, pos_emb, p=2)
    neg_dist = F.pairwise_distance(anchor_emb, neg_emb, p=2)

    loss = F.relu(pos_dist - neg_dist + margin)
    return loss.mean()

def orthogonality_loss (t): 
    eye = torch.eye(t.shape[0], device=t.device) 
    return ((t @t.T - eye)**2).sum()


def distillation_loss(student_logits, teacher_logits, T=1.5):
    p_teacher = F.softmax(teacher_logits / T, dim=1).clamp(min=1e-8)
    p_student = F.log_softmax(student_logits / T, dim=1)
    kl_loss = F.kl_div(p_student, p_teacher, reduction='batchmean') * (T * T)
    return kl_loss

def ppv_sensitivity(specificity_levels, _y_true, _y_pred_proba):
    add_sensitivity_results = []
    add_ppv_results =  []
    add_fpr, add_tpr, add_thresholds = roc_curve(_y_true, _y_pred_proba)
    _results = {}
    for specificity in specificity_levels:

        _threshold_index = np.where(add_fpr <= (1 - specificity))[0][-1]
        _threshold = add_thresholds[_threshold_index]

        _sensitivity = add_tpr[_threshold_index]
        add_sensitivity_results.append( _sensitivity)

        _y_pred_binary = (_y_pred_proba >= _threshold).astype(int)
        _ppv = precision_score(_y_true, _y_pred_binary)
        add_ppv_results.append(_ppv)

    _results['Sensitivity'] = add_sensitivity_results
    _results['PPV'] = add_ppv_results

    return _results

from typing import Callable
def compute_triplet_loss_with_scorer(
    node_emb: torch.Tensor,
    triplet_index: torch.LongTensor,
    scorer: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    margin: float = 1.0
) -> torch.Tensor:

    anc_idx, pos_idx, neg_idx = triplet_index
    anchor = node_emb[anc_idx]  
    pos = node_emb[pos_idx]     
    neg = node_emb[neg_idx]     

    s_pos = scorer(anchor, pos)  
    s_neg = scorer(anchor, neg)  

    loss = F.relu(s_neg - s_pos + margin)
    return loss.mean()




