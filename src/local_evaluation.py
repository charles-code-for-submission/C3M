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

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

def set_seed_all(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed_all(42)

def evaluate_model( model, studentmodel, val_loader, criterion, common_ids, dict_ehr, dict_gene, genetic_matrix, device='cpu', graph=None):
    model.eval()  
    studentmodel.eval()
    y_true = []
    y_scores = []
    y_preds = []
    total_loss = 0.0  
    
    with torch.no_grad():  
        for batch in val_loader:
            batch_y = batch['label'].to(device)  
            batch_id = batch['patient_id']
            batch_x = batch['ehr_input'].to(device)  

            common_ids_batch = batch_id
            test_mask = torch.tensor([bi in common_ids_batch for bi in batch_id])
            test_y = batch_y[test_mask]
            test_ehr0 = batch_x[test_mask]

            tindex = True
            if tindex is True:
                test_id = [tid for m, tid in zip(test_mask, batch_id) if m == True ]
                tid_index  = np.array([dict_ehr[tid] for tid in test_id])
            else:
                tid_index = None
            test_ehr_g = model.ehr_emb2( graph.edge_index)[tid_index]
            test_y_pred_gene, _ = studentmodel(test_ehr_g)
            test_loss_gene = criterion(test_y_pred_gene, test_y)

            probabilities = torch.softmax(test_y_pred_gene, dim=1)[:, 1]
            y_pred = torch.argmax(test_y_pred_gene, dim=1) 

            y_true.extend(test_y.cpu().numpy()) 

            y_scores.extend(probabilities.cpu().numpy()) 
            y_preds.extend(y_pred.cpu().numpy())

            total_loss += test_loss_gene.item() * test_y.size(0) 

    average_loss = total_loss / len(y_true)

    auroc = roc_auc_score(y_true, y_scores)
    acc = accuracy_score(y_true, y_preds)   
    f1_s = f1_score(y_true, y_preds, average='binary')
       
    add_results = ppv_sensitivity([0.9, 0.95], y_true, y_scores)
    sensitivity_90, sensitivity_95 = add_results['Sensitivity']
    ppv_90, ppv_95 = add_results['PPV']
    
    auprc = average_precision_score(y_true, y_scores)

    return acc, f1_s, auroc, auprc, sensitivity_90, sensitivity_95, ppv_90, ppv_95, average_loss

def local_finetuning(f_feature_matrix, feature_matrix,  label_array, id_array, genetic_matrix, n_splits=5, batch_size=64,\
                           num_steps=1000, dict_ehr=None, dict_gene =None, common_ids=None, \
                           dropout=0, lr=0.001, ehr_hidden_dims=[64, 32], pred_input_dim=64,\
                           gnn_channel=32, attention_out_dim=32, device=None,state_dict=None,\
                                state_dict_student=None,samplesize=None):
    
    
    ehr_dim = feature_matrix.shape[1]

    init_gene_unique=None
    for fold, train_loader, test_loader, tGraph, _ in create_dataloaders(f_feature_matrix, feature_matrix, label_array,\
                                                                 id_array, n_splits, batch_size, samplesize=samplesize):
        print('Train loader: ', len(train_loader), '| Val loader: ' , len(test_loader))

        ehrgraph = tGraph.data
        num_nodes = ehrgraph.x.size(0)
        ei = ehrgraph.edge_index
        assert ei.min() >= 0,            "negative index in edge_index!"
        assert ei.max() < num_nodes,     f"edge_index.max()={ei.max()} but num_nodes={num_nodes}"
        ehrgraph = ehrgraph.to(device)
        f_feature_matrix = torch.tensor(f_feature_matrix, device=device)
        feature_matrix = torch.tensor(feature_matrix, dtype=torch.float32, device=device)
                
        model =  EHREncoder(tGraph, in_channels=gnn_channel, hidden_channels=gnn_channel, out_channels=gnn_channel, dropout=dropout,\
                 eid_emb=feature_matrix,  ehr_dim=ehr_dim, ehr_hidden_dims=ehr_hidden_dims ).to(device)
        
        studentmodel = StudentModel().to(device)
        
        optimizer = torch.optim.Adam(
            list(model.parameters())+list(studentmodel.parameters()), 
            lr=lr,
            eps= 1e-5,
        )

        if state_dict is not None:
            cur_model_dict = model.state_dict()
            cur_model_dict_student = studentmodel.state_dict()
            print("Model parameters loaded successfully.")


            for _name, _param in state_dict.items():
                print(f"Loading {_name}", _param.shape)
                cur_model_dict[_name].copy_(_param)
            
            for _name, _param in state_dict_student.items():
                cur_model_dict_student[_name].copy_(_param)

        else:
            print("Model from stratch.")



        class_weights = torch.tensor([1.0, 10.0], device=device)

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

        curve_losses = {}
        for epoch in range(num_steps):
            print(f"\nEpoch {epoch}/{num_steps}...")
            model.train()  
            studentmodel.train()

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
                    common_ids_batch = batch_id     
                    test_mask = torch.tensor([bi in common_ids_batch for bi in batch_id])
                    test_y = batch_y[test_mask] 
                    tindex = True
                    if tindex is True:
                        test_id = [tid for m, tid in zip(test_mask, batch_id) if m == True ]
                        tid_index  = np.array([dict_ehr[tid] for tid in test_id])
                    else:
                        tid_index = None

                    test_ehr_g = model.ehr_emb2(ehrgraph.edge_index) 
                    test_ehr_g = test_ehr_g[tid_index]
                    test_y_pred_gene, _ = studentmodel(test_ehr_g)
                    loss_main = criterion(test_y_pred_gene, test_y)

                total_loss = loss_main
                optimizer.zero_grad()
                total_loss.backward()

                if epoch % 201 == 0:
                    for name, param in model.named_parameters():
                        print(f'{name} {param.shape}, grad: {param.grad}')

                optimizer.step()
                epoch_loss += total_loss.item()
                epoch_loss_main += loss_main.item()
                epoch_loss_gene += loss_genetic.item()
                epoch_loss_ortho += loss_ortho.item()


            avg_epoch_loss = epoch_loss / len(train_loader)
            avg_main_loss = epoch_loss_main / len(train_loader)
            avg_gene_loss = epoch_loss_gene / len(train_loader)
            avg_ortho_loss = epoch_loss_ortho / len(train_loader)

            curve_losses.setdefault('train_main', []).append(avg_main_loss)
            curve_losses.setdefault('train_gene', []).append(avg_gene_loss)
            curve_losses.setdefault('train_ortho', []).append(avg_ortho_loss)

            if epoch % 1 == 0:
                print(f"Epoch {epoch} | Train loss: {avg_epoch_loss:.4f} | Train main loss: {avg_main_loss:.4f} | Train gene loss: {avg_gene_loss:.4f}| Train ortho loss: {avg_ortho_loss:.4f}")
                
            if epoch % 1 == 0:   
                acc, f1binary, auroc, auprc, sensi90, sensi95, ppv90, ppv95, average_loss = evaluate_model\
                ( model, studentmodel, test_loader, criterion, common_ids, dict_ehr, dict_gene, None, device=device, graph=ehrgraph)
                
                print(f"\tFold {fold} | Step {step} | Validation Loss: {average_loss:.4f}")
                print(f"\tFold {fold} | Step {step} | AUROC: {auroc:.4f}")
                print(f"\tFold {fold} | Step {step} | auprc: {auprc:.4f}")
                print(f"\tFold {fold} | Step {step} | ACC: {acc:.4f}")
                print(f"\tFold {fold} | Step {step} | F1binary: {f1binary:.4f}")

                print(f"\tFold {fold} | Step {step} | sensi90: {sensi90:.4f}")
                print(f"\tFold {fold} | Step {step} | ssensi95: {sensi95:.4f}")
                print(f"\tFold {fold} | Step {step} | ppv90: {ppv90:.4f}")
                print(f"\tFold {fold} | Step {step} | ppv95: {ppv95:.4f}")

            curve_losses.setdefault('val_loss', []).append(average_loss)
            curve_losses.setdefault('val_auroc', []).append(auroc)
            curve_losses.setdefault('val_auprc', []).append(auprc)
            curve_losses.setdefault('val_acc', []).append(acc)
            curve_losses.setdefault('val_f1binary', []).append(f1binary)

            curve_losses.setdefault('val_sensi90', []).append(sensi90)
            curve_losses.setdefault('val_sensi95', []).append(sensi95)
            curve_losses.setdefault('val_ppv90', []).append(ppv90)
            curve_losses.setdefault('val_ppv95', []).append(ppv95)

            if  auroc> best_metrics['auroc']:
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
                torch.save(model.state_dict(), f'model_ckpt/model_best_fold{fold}.pt')
                counter = 0  # reset counter on improvement
                print(f" New best model at epoch {epoch}, auroc: {auroc:.4f}")

            else:
                counter += 1
                print(f" No improvement. Early stop counter: {counter}/{patience}, Best model at epoch {best_metrics['epoch']}, auroc: {best_metrics['auroc']:.4f}")
                if counter >= patience:
                    print(" Early stopping triggered.")
                    metrics_to_print = ['epoch', 'auroc', 'auprc', 'acc', 'f1', 'sensi90', 'sensi95', 'ppv90', 'ppv95', 'loss']

                    for metric in metrics_to_print:
                        values =f"{best_metrics[metric]:.4f}"
                        print(f'-> {metric} | {values}')
                    break

        print('\n')
        metrics_to_print = ['epoch', 'auroc', 'auprc', 'acc', 'f1', 'sensi90', 'sensi95', 'ppv90', 'ppv95', 'loss']
        for metric in metrics_to_print:
            values =f"{best_metrics[metric]:.4f}"
            print(f'-> {metric} | {values}')
        break
    return curve_losses



def prepare_data(wgene=False, year=1, reference_columns=None):
    if wgene:
        final_gene_select = pickle.load(open('Data/final_gene_select.pkl', 'rb'))
        snp_gene_nx_graph = pickle.load(open('Data/snp_gene_nx_graph.pkl', 'rb'))

        gene_ids_list = final_gene_select.s.values.tolist()
        gene_fea_for_certain_year = final_gene_select.iloc[:, 1:].to_numpy()
        gene_ids_for_certain_year = [str(i) for i in gene_ids_list]
        print(len(gene_ids_for_certain_year), gene_fea_for_certain_year.shape)

    motorpath1 = f"NewHome/motor_dir/motor_reprs_{year}.pkl"
    motorpath2 = f"NewHome/motor_dir/trash/year_{year}/patients_dict.pkl"

    motor_rep = pickle.load(open(motorpath1, 'rb'))
    input_ehr_emb = motor_rep['representations']
    print('| input_ehr_emb shape:', input_ehr_emb.shape)

    pindex2pid = pickle.load(open(motorpath2, 'rb'))
    rev_pindex2pid = {v:k for k,v in pindex2pid.items()}
    patients_for_certain_year = motor_rep['patient_ids']
    motor_patients_for_certain_year_str = [rev_pindex2pid[i] for i in patients_for_certain_year]

    print('| dict of pid:', len(rev_pindex2pid))
    assert len(rev_pindex2pid) == len(motor_patients_for_certain_year_str) , f"dict differ"
    
    f = pickle.load(open('EHR_DATA/feature_files/matched_f_features.pkl', 'rb'))
    t = pickle.load(open('EHR_DATA/feature_files/adrd_y.pkl', 'rb'))
    f_this_year = f[year]
    t_this_year = t[year]

    fcols = [c for c in f_this_year.columns if c.endswith('_rx') or c.endswith('_dx') or c == 'person_id' or c=='HASH_SUBJECT_ID']
    print('| f_this_year columns:', len(f_this_year.columns), '; all kept columns:' , len(fcols))
    f_this_year['HASH_SUBJECT_ID'] = f_this_year['HASH_SUBJECT_ID'].astype(str)
    f_this_year = f_this_year[fcols]
    print(f'--- make same f features as source domain', 'green')

    ref_cols = [_c for _c in reference_columns if _c != 'person_id']
    for _col in ref_cols:
        if _col not in f_this_year.columns:
            f_this_year.loc[:,_col] = 0
    ref_cols = ['HASH_SUBJECT_ID'] + ref_cols
    f_this_year = f_this_year[ref_cols]
    assert f_this_year.shape[-1] == len(ref_cols)
    f_this_year_ehr = f_this_year.iloc[:, 1:].to_numpy()
    print('| f_this_year_ehr numpy:', f_this_year_ehr.shape)

    ehr_ori_ids = f_this_year['HASH_SUBJECT_ID'].astype(str).values.tolist()
    print('| ids from one-hot feature, mapped', len(ehr_ori_ids))
    both_ids1 = [i for i in ehr_ori_ids if i in motor_patients_for_certain_year_str] 
    print('| ids in both motor and one-hot data', len(both_ids1))
    both_ids = [i for i in motor_patients_for_certain_year_str if i in both_ids1]
    print('| ids in both motor and one-hot data', len(both_ids1))

    both_ids_index_in_motor = np.array([i in both_ids for i in motor_patients_for_certain_year_str]).astype(int)
    print('| both_ids_index_in_motor', both_ids_index_in_motor.shape, both_ids_index_in_motor.sum())
    both_ids_index_in_original_ehr = np.array([i in both_ids for i in ehr_ori_ids]).astype(int)
    print('| both_ids_index_in_original_ehr', both_ids_index_in_original_ehr.shape, both_ids_index_in_original_ehr.sum())
    
    _ehr_ori_ids = [i for i, j in zip(ehr_ori_ids, both_ids_index_in_original_ehr) if j==1]
    _ehr_ori_ids = [pindex2pid[id] for id in _ehr_ori_ids]
    _ehr_map_index = [i - 1 for i in _ehr_ori_ids]
    print('| _ehr_map_index for motor', len(_ehr_map_index))

    adj_input_ehr_emb = input_ehr_emb[_ehr_map_index]
    motor_patients_for_certain_year = patients_for_certain_year[_ehr_map_index]
    assert (_ehr_ori_ids == motor_patients_for_certain_year).all(), f"Lists differ"

    adj_input_ehr_y = t_this_year[both_ids_index_in_original_ehr==1]
    adj_f_this_year_ehr = f_this_year_ehr[both_ids_index_in_original_ehr==1]
    

    adj_patients_for_certain_year = both_ids
    if wgene:
        print('adj_patients_for_certain_year', len(adj_patients_for_certain_year))

        _ids_index_in_gene = np.array([i in adj_patients_for_certain_year for i in gene_ids_for_certain_year]).astype(int)

        adj_gene_ids_for_certain_year = [ id for i, id in enumerate(gene_ids_for_certain_year) if _ids_index_in_gene[i]==1] 

        adj_gene_fea_for_certain_year = gene_fea_for_certain_year[_ids_index_in_gene==1]
        print('adj_gene_fea_for_certain_year', adj_gene_fea_for_certain_year.shape, len(adj_gene_ids_for_certain_year))
    else:
        adj_gene_ids_for_certain_year, adj_gene_fea_for_certain_year = None, None

    return adj_patients_for_certain_year, adj_gene_ids_for_certain_year, adj_f_this_year_ehr, adj_input_ehr_emb, adj_input_ehr_y, adj_gene_fea_for_certain_year


def _main(stact_dict, state_dict_student, reference_columns):

    adj_patients_for_certain_year, adj_gene_ids_for_certain_year, adj_f_this_year_ehr, \
        adj_input_ehr_emb, adj_input_ehr_y, adj_gene_fea_for_certain_year = prepare_data(wgene=False, reference_columns=reference_columns)
    
    assert type(adj_patients_for_certain_year[0]) == str
    if adj_gene_ids_for_certain_year is not None:
        assert type(adj_gene_ids_for_certain_year[0]) == str

    ehr_index_dict, gene_index_dict, common_ids = get_patient_indices(adj_patients_for_certain_year, \
                                                                    adj_gene_ids_for_certain_year,\
                                                                    adj_patients_for_certain_year) 

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)


    curve_for_plot = local_finetuning(adj_f_this_year_ehr, adj_input_ehr_emb, adj_input_ehr_y, adj_patients_for_certain_year, adj_gene_fea_for_certain_year, \
                        n_splits=5, batch_size=1024, num_steps=1000, dict_ehr=ehr_index_dict,\
                        dict_gene=gene_index_dict, common_ids=common_ids, dropout=0.1, \
                        lr=0.001, ehr_hidden_dims=[64, 16], pred_input_dim=[16, 16], \
                        gnn_channel=16, attention_out_dim=16, state_dict=stact_dict, \
                                state_dict_student=state_dict_student, samplesize=500) 
    
    return curve_for_plot

state_dict = torch.load('pretrainedModels/model_best_fold0.pt', map_location='cpu')
state_dict_student = torch.load('pretrainedModels/model_best_fold0.pt', map_location='cpu')  
reference_columns = pickle.load(open('pretrainedModels/reference_columns.pkl', 'rb'))
curve_for_plot = _main(state_dict, state_dict_student, reference_columns)
