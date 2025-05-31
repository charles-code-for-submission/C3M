import argparse
import os
import random 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from utils import *
from utils_data import *
from models import *

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def evaluate_model( model, student, val_loader, criterion, common_ids, dict_ehr, dict_gene, genetic_matrix, device='cpu', graph=None):
    model.eval()  
    y_true = []
    y_scores = []
    y_preds = []
    total_loss = 0.0  
    
    with torch.no_grad():
        for batch in val_loader:
            batch_y = batch['label'].to(device)  
            batch_id = batch['patient_id']
            batch_x = batch['ehr_input'].to(device)  

            common_ids_batch = [bid for bid in batch_id if bid in common_ids ] 

            test_mask = torch.tensor([bi in common_ids_batch for bi in batch_id])
            test_y = batch_y[test_mask]
            test_ehr0 = batch_x[test_mask]
            test_ehr_e = model.ehr_emb(test_ehr0)

            tindex = True
            if tindex is True:
                test_id = [tid for m, tid in zip(test_mask, batch_id) if m == True ]
                tid_index  = np.array([dict_ehr[tid] for tid in test_id])
            else:
                tid_index = None
            test_ehr_g = model.ehr_emb2( graph.edge_index)[tid_index]
            person_aft_gene_bank = test_ehr_g
            test_y_pred_gene, _ = student(person_aft_gene_bank)

            test_loss_gene = criterion(test_y_pred_gene, test_y)

            probabilities = torch.softmax(test_y_pred_gene, dim=1)[:, 1]  
            y_pred = torch.argmax(test_y_pred_gene, dim=1) 

            y_true.extend(test_y.cpu().numpy()) 

            y_scores.extend(probabilities.cpu().numpy())
            y_preds.extend(y_pred.cpu().numpy())

            total_loss += test_loss_gene.item() * test_y.size(0) 

    average_loss = total_loss / len(y_true)

    auroc = roc_auc_score(y_true, y_scores)
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score

    acc = accuracy_score(y_true, y_preds)   
    f1_s = f1_score(y_true, y_preds, average='binary')
       
    add_results = ppv_sensitivity([0.9, 0.95], y_true, y_scores)
    sensitivity_90, sensitivity_95 = add_results['Sensitivity']
    ppv_90, ppv_95 = add_results['PPV']
    
    auprc = average_precision_score(y_true, y_scores)

    return acc, f1_s, auroc, auprc, sensitivity_90, sensitivity_95, ppv_90, ppv_95, average_loss


def train_with_genetics( f_feature_matrix, feature_matrix,  label_array, id_array, genetic_matrix, n_splits=5, batch_size=64,\
                           num_steps=1000, dict_ehr=None, dict_gene =None, common_ids=None, \
                           dropout=0, lr=0.001, ehr_hidden_dims=[64, 32], pred_input_dim=64,\
                           gnn_channel=32, attention_out_dim=32, lambda_stu=0.01):
     
    genetic_dim = genetic_matrix.shape[1]
    ehr_dim = feature_matrix.shape[1]
    f_ehr_dim = f_feature_matrix.shape[1]

    all_ids = len(id_array)
    print('number of (ehr and gene) common ids:', len(common_ids))
    index_of_id_who_has_genetic_data = [i in common_ids for i in id_array]
    modified_genetic_matrix = np.zeros((all_ids, genetic_dim))
    modified_genetic_matrix[index_of_id_who_has_genetic_data, :] = genetic_matrix

    for fold, train_loader, val_loader, _,  tGraph, _ in create_ehrgraph_dataloaders(f_feature_matrix, feature_matrix, label_array,\
                                                                 id_array, n_splits, batch_size):
        print(f"Fold {fold} training...")
        print('Train loader: ', len(train_loader), '| Val loader: ' , len(val_loader))

        ehrgraph = tGraph.data
        num_nodes = ehrgraph.x.size(0)
        ei = ehrgraph.edge_index
        assert ei.min() >= 0,            "negative index in edge_index!"
        assert ei.max() < num_nodes,     f"edge_index.max()={ei.max()} but num_nodes={num_nodes}"
        ehrgraph = ehrgraph.to(device)
        
        '''modify input properly'''
        f_feature_matrix = torch.tensor(f_feature_matrix, device=device)
        feature_matrix = torch.tensor(feature_matrix, dtype=torch.float32, device=device)
                            
        model =  EHREncoder(tGraph, in_channels=gnn_channel, hidden_channels=gnn_channel, out_channels=gnn_channel, dropout=dropout,\
                 eid_emb=feature_matrix,  ehr_dim=ehr_dim, ehr_hidden_dims=ehr_hidden_dims).to(device)
        
        teachermodel = TeacherTransformerModel().to(device)
        studentmodel = StudentModel().to(device)

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(teachermodel.parameters()), 
            lr=lr,
            eps= 1e-5,
        )

        soptimizer = torch.optim.Adam(
            studentmodel.parameters(), 
            lr=0.0001,
            eps= 1e-5,
        )

        for name, param in model.named_parameters():
            print(f'{name} {param.shape}, requires grad: {param.requires_grad}')
            
        class_weights = torch.tensor([1.0, 10.0], device=device)  # 少数类（类别1）权重更高

        criterion = nn.CrossEntropyLoss(weight=class_weights)

        print('\n--------------------Start training-------------------')


        best_metrics = {
            'epoch': -1,
            'auroc': 0,
            'acc': 0,
            'f1': 0,
            'loss': float('inf'),
            'auprc': 0,
            'sensi90': 0,
            'sensi95': 0,
            'ppv90': 0,
            'ppv95': 0
        }
        patience = 300
        counter = 0

        for epoch in range(num_steps):
            print(f"\nEpoch {epoch}/{num_steps}...")
            
            epoch_loss = 0
            epoch_loss_main = 0
            epoch_loss_gene = 0
            epoch_loss_ortho = 0
            loss_genetic = torch.tensor(0.0)
            loss_ortho = torch.tensor(0.0)
            for step, ehr_batch in enumerate(train_loader):
                batch_id = ehr_batch['patient_id']   
                batch_y = ehr_batch['label'].to(device)  
                batch_x = ehr_batch['ehr_input'].to(device)  

                if step % 1 == 0:
                    model.train() 
                    teachermodel.train()

                    common_ids_batch = [bid for bid in batch_id if bid in common_ids]
                    gene_batch_index  = np.array([dict_gene[i] for i in common_ids_batch])
                    gene_x =  torch.tensor(genetic_matrix[gene_batch_index],dtype=torch.float32).to(device)
                    gene_x = (gene_x > 0).float()
                    
                    test_mask = torch.tensor([bi in common_ids_batch for bi in batch_id])
                    test_y = batch_y[test_mask] 

                    tindex = True
                    if tindex is True:
                        test_id = [tid for m, tid in zip(test_mask, batch_id) if m == True]
                        tid_index  = np.array([dict_ehr[tid] for tid in test_id])
                    else:
                        tid_index = None

                    test_ehr_g = model.ehr_emb2(ehrgraph.edge_index)
                    test_ehr_g = test_ehr_g[tid_index]

                    person_aft_gene_bank = test_ehr_g

                    test_y_pred_gene, th, rec_logits, mask_scores = teachermodel(person_aft_gene_bank, gene_x)

                    loss_main = criterion(test_y_pred_gene, test_y)
                    B, N = gene_x.shape
                    rec_flat = rec_logits.flatten(0,1)
                    tgt_flat = gene_x.flatten().long()  
                    w = mask_scores.view(B*N).detach()
                    loss_rec = F.cross_entropy(rec_flat, tgt_flat, reduction='none')
                    loss_rec = (loss_rec * w).sum() / w.sum()
                    total_loss = loss_main + 0.1*loss_rec   #0.1

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    teachermodel.eval()
                    model.eval()

                    with torch.no_grad():
                        test_y_pred_gene_s, th_s, _, _ = teachermodel(person_aft_gene_bank, gene_x)

                    studentmodel.train()
                    s_logit, sh = studentmodel(person_aft_gene_bank.detach())
                    s_loss = distillation_loss(s_logit, test_y_pred_gene_s) + lambda_stu * criterion(s_logit, test_y)

                    soptimizer.zero_grad()
                    s_loss.backward()
                    soptimizer.step()

                # if epoch % 300 == 0:
                #     for name, param in model.named_parameters():
                #         print(f'{name} {param.shape}, grad: {param.grad}')

                epoch_loss += total_loss.item()
                epoch_loss_main += loss_main.item()
                epoch_loss_gene += loss_genetic.item()
                epoch_loss_ortho += loss_ortho.item()

            avg_epoch_loss = epoch_loss / len(train_loader)
            avg_main_loss = epoch_loss_main / len(train_loader)
            avg_gene_loss = epoch_loss_gene / len(train_loader)
            avg_ortho_loss = epoch_loss_ortho / len(train_loader)

            if epoch % 2 == 0:
                print(f"Epoch {epoch} | Train loss: {avg_epoch_loss:.4f} | Train main loss: {avg_main_loss:.4f} | Train gene loss: {avg_gene_loss:.4f}| Train ortho loss: {avg_ortho_loss:.4f}")
                
            if epoch % 2 == 0:   
                acc, f1binary, auroc, auprc, sensi90, sensi95, ppv90, ppv95,average_loss = evaluate_model\
                ( model, studentmodel, val_loader, criterion, common_ids, dict_ehr, dict_gene, modified_genetic_matrix, device=device, graph=ehrgraph)
                
                print(f"\tFold {fold} | Step {step} | Validation Loss: {average_loss:.4f}")
                print(f"\tFold {fold} | Step {step} | AUROC: {auroc:.4f}")
                print(f"\tFold {fold} | Step {step} | auprc: {auprc:.4f}")
                print(f"\tFold {fold} | Step {step} | ACC: {acc:.4f}")
                print(f"\tFold {fold} | Step {step} | F1binary: {f1binary:.4f}")

                print(f"\tFold {fold} | Step {step} | sensi90: {sensi90:.4f}")
                print(f"\tFold {fold} | Step {step} | ssensi95: {sensi95:.4f}")
                print(f"\tFold {fold} | Step {step} | ppv90: {ppv90:.4f}")
                print(f"\tFold {fold} | Step {step} | ppv95: {ppv95:.4f}")

            if auroc > best_metrics['auroc']:
                best_metrics.update({
                    'epoch': epoch,
                    'auroc': auroc,
                    'acc': acc,
                    'f1': f1binary,
                    'loss': average_loss,
                    'auprc': auprc,
                    'sensi90': sensi90,
                    'sensi95': sensi95,
                    'ppv90': ppv90,
                    'ppv95': ppv95,
                })
                torch.save(model.state_dict(), f'model_ckpt_sc/model_best_fold{fold}_transformer-4.pt')
                torch.save(studentmodel.state_dict(), f'model_ckpt_sc/model_best_fold{fold}_student_transformer-4.pt')
                counter = 0  # reset counter on improvement
                print(f"New best model at epoch {epoch}, AUROC: {auroc:.4f}")
            else:
                counter += 1
                print(f"No improvement. Early stop counter: {counter}/{patience}")
                print(f"-- best model at epoch {best_metrics['epoch']}, AUROC: {best_metrics['auroc']:.4f}")

                if counter >= patience:
                    print("Early stopping triggered.")

                    metrics_to_print = ['epoch', 'auroc', 'auprc', 'acc', 'f1', 'sensi90', 'sensi95', 'ppv90', 'ppv95', 'loss']

                    for metric in metrics_to_print:
                        values =f"{best_metrics[metric]:.4f}"
                        print(f'-> {metric} | {values}')
                        # print(f"{metric}:\t" + "\t".join(values))
                    break

        break
    

def assert_sorted(lst):
    assert lst == sorted(lst), f"List is not sorted: {lst}"


def parse_arg():
    parser = argparse.ArgumentParser(
        description="Example script demonstrating argparse"
    )

    parser.add_argument(
        '--lr', 
        type=float,
        default=0.001,
    )    
    
    parser.add_argument(
        '--weight_genetic',
        type=float,
        default=0.01,
    )

    parser.add_argument(
        '--weight_ehr',
        type=float,
        default=0.01,
    )

    parser.add_argument(
        '--cuda',
        type=int,
        default=2,
    )

    parser.add_argument(
        '--glarge',
        type=bool,
        default=True,
    )

    parser.add_argument(
        '--patience',
        type=int,
        default=300,
    )

    parser.add_argument(
        '--gnn_dim',
        type=int,
        default=16,
    )

    parser.add_argument(
        '--att_dim',
        type=int,
        default=16,
    )

    parser.add_argument(
        '--bank_dim',
        type=int,
        default=16,
    )


    parser.add_argument(
        '--note',
        type=str,
        default='',
    )

    parser.add_argument(
        '--lambda_stu',
        type=float,
        default=0.01,
    )


    parser.add_argument(
        '--seed',
        type=int,
        default=42,
    )

    args = parser.parse_args()
    return args

arg_values = parse_arg()

device = torch.device(f'cuda:{arg_values.cuda}' if torch.cuda.is_available() else 'cpu')
print('\nUsing device:', device)

adj_patients_for_certain_year, adj_gene_ids_for_certain_year, adj_f_this_year_ehr, \
    adj_input_ehr_emb, adj_input_ehr_y, adj_gene_fea_for_certain_year = prepare_data('aou')

import random
assert type(adj_patients_for_certain_year[0]) == str
assert type(adj_gene_ids_for_certain_year[0]) == str


num_classes = 2  
set_seed(arg_values.seed)

ehr_index_dict, gene_index_dict, common_ids = get_patient_indices(adj_patients_for_certain_year, \
                                                                adj_gene_ids_for_certain_year,\
                                                                adj_patients_for_certain_year) 

assert_sorted(adj_gene_ids_for_certain_year)
assert_sorted(adj_patients_for_certain_year)

train_with_genetics(adj_f_this_year_ehr, adj_input_ehr_emb, adj_input_ehr_y, adj_patients_for_certain_year, adj_gene_fea_for_certain_year, \
                    n_splits=5, batch_size=512, num_steps=2500, dict_ehr=ehr_index_dict,\
                    dict_gene=gene_index_dict, common_ids=common_ids, dropout=0.1, \
                    lr=arg_values.lr, ehr_hidden_dims=[64, 16],\
                    pred_input_dim=[arg_values.gnn_dim, 16], \
                    gnn_channel=arg_values.gnn_dim, attention_out_dim=arg_values.att_dim, lambda_stu=arg_values.lambda_stu) 

