import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, DataLoader as GeoDataLoader


def create_ehrgraph_dataloaders(f_feature_matrix, feature_matrix, label_array, id_array, n_splits=5, batch_size=64, \
                           genetic_matrix=None, dict_gene={}, common_ids=[], device='cpu'):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42,val_ratio=0.2)
    X = feature_matrix

    y_split =  label_array 
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y_split)):

        X_trainval = X[train_idx]
        y_trainval = label_array[train_idx]
        id_trainval = [id_array[i] for i in train_idx.tolist()]

        X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
            X_trainval, y_trainval, id_trainval, test_size=val_ratio, stratify=y_trainval, random_state=42)

        X_test = X[test_idx]
        y_test = label_array[test_idx]
        id_test = [id_array[i] for i in test_idx.tolist()]

        print('build ehr graph,', f_feature_matrix.shape)
        ehr_graph = build_ehr_nx_graph(range(f_feature_matrix.shape[0]), f_feature_matrix)  # train id, train id with gene, subset of both train and both having-gene ids

        train_gDataset = GraphEHRDataset(ehr_graph)
        
        train_dataset = EHRDataset(X_train, id_train, y_train)  # based on feature_values, patient_ids, label_array):
        test_dataset = EHRDataset(X_test, id_test, y_test) 
        val_dataset = EHRDataset(X_val, id_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1) 
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
        yield fold, train_loader, test_loader, train_gDataset, None



def create_ehr_dataloaders(feature_matrix, label_array, id_array, n_splits=5, batch_size=64, \
                           genetic_matrix=None, dict_gene={}, common_ids=[], device='cpu'):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42, val_ratio=0.33)
    X = feature_matrix

    y_split =  label_array 
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y_split)):

        X_trainval = X[train_idx]
        y_trainval = label_array[train_idx]
        id_trainval = [id_array[i] for i in train_idx.tolist()]

        X_train, X_val, y_train, y_val, id_train, id_val = train_test_split(
            X_trainval, y_trainval, id_trainval, test_size=val_ratio, stratify=y_trainval, random_state=42)

        X_test = X[test_idx]
        y_test = label_array[test_idx]
        id_test = [id_array[i] for i in test_idx.tolist()]

        common_ids_tr = [bid for bid in id_train if bid in common_ids ] #list(set(batch_id) & set(common_ids))
        gene_tr_index  = np.array([dict_gene[i] for i in common_ids_tr])
        gene_tr_x =   genetic_matrix[gene_tr_index] 
        nxg = build_nx_graph(common_ids_tr, gene_tr_x) 
        Ginfo(nxg)

        train_gDataset = GraphDataset(nxg)
        
        train_dataset = EHRDataset(X_train, id_train, y_train)  # based on feature_values, patient_ids, label_array):
        test_dataset = EHRDataset(X_test, id_test, y_test) 
        val_dataset = EHRDataset(X_val, id_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) 
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        yield fold, train_loader, val_loader, test_loader, train_gDataset, common_ids_tr



def prepare_data(foldername):

    input_ehr_emb = pickle.load( open('Data/input_ehr_features.pkl', 'rb'))
    input_ehr_y = pickle.load( open('Data/input_ehr_y.pkl', 'rb'))
    patients_for_certain_year = pickle.load(open('Data/patients_for_certain_year.pkl', 'rb'))

    final_gene = pickle.load(open('Data/final_gene_merge.pkl', 'rb'))
    gene_fea_for_certain_year = final_gene.iloc[:, 1:].to_numpy()
    gene_ids_for_certain_year = [str(i) for i in final_gene.s.values.tolist()]


    f = load_all_pickle_chunks(f'Data/{foldername}.pkl')
    t = pickle.load( open('Data/t_1year.pkl','rb'))
    e = pickle.load( open('Data/e_1year.pkl','rb'))
    f_this_year = f


    fcols = [c for c in f_this_year.columns if c.endswith('_rx') or c.endswith('_dx') or c == 'person_id']
    print(len(f_this_year.columns), len(fcols))
    f_this_year['person_id'] = f_this_year['person_id'].astype(str)
    f_this_year = f_this_year[fcols]
    print(f_this_year.shape)

    f_this_year_ehr = f_this_year.iloc[:, 1:].to_numpy()
    print(f_this_year_ehr.shape)

    ehr_ori_ids = f_this_year['person_id'].astype(str).values.tolist()
    print(len(ehr_ori_ids))
    print(len(patients_for_certain_year))
    both_ids1 = [i for i in ehr_ori_ids if i in patients_for_certain_year] 
    print(len(both_ids1))
    both_ids = [i for i in patients_for_certain_year if i in both_ids1]
    print(len(both_ids))

    both_ids_index_in_motor = np.array([i in both_ids for i in patients_for_certain_year]).astype(int)
    print(both_ids_index_in_motor.shape, both_ids_index_in_motor.sum())

    both_ids_index_in_original_ehr = np.array([i in both_ids for i in f_this_year['person_id'].astype(str).values.tolist()]).astype(int)
    print(both_ids_index_in_original_ehr.shape, both_ids_index_in_original_ehr.sum())


    adj_input_ehr_emb = input_ehr_emb[both_ids_index_in_motor==1]
    adj_input_ehr_y = input_ehr_y[both_ids_index_in_motor==1]

    adj_f_this_year_ehr = f_this_year_ehr[both_ids_index_in_original_ehr==1]
    adj_t_this_year_ehr = t['CP_1_1_yr'][both_ids_index_in_original_ehr==1]

    adj_patients_for_certain_year = both_ids

    _ids_index_in_gene = np.array([i in adj_patients_for_certain_year for i in gene_ids_for_certain_year]).astype(int)

    adj_gene_ids_for_certain_year = [ id for i, id in enumerate(gene_ids_for_certain_year) if _ids_index_in_gene[i]==1] 

    adj_gene_fea_for_certain_year = gene_fea_for_certain_year[_ids_index_in_gene==1]

    return adj_patients_for_certain_year, adj_gene_ids_for_certain_year, adj_f_this_year_ehr, adj_input_ehr_emb, adj_input_ehr_y, adj_gene_fea_for_certain_year


def create_dataloaders(f_feature_matrix, feature_matrix, label_array, id_array, n_splits=5, batch_size=64, \
                           genetic_matrix=None, dict_gene={}, common_ids=[], device='cpu', samplesize=None):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42, samplesize=500)
    X = feature_matrix

    y_split =  label_array 
    fewshot = True
    if fewshot:
        train_idx = np.random.choice(len(X), size=samplesize, replace=False)
        test_idx = np.setdiff1d(np.arange(len(X)), train_idx)
        fold = 0
    else:
        for fold, (train_idx, test_idx) in enumerate(kf.split(X, y_split)):
            if fold ==0:
                print('fold', train_idx[:5], test_idx[:5], len(train_idx), len(test_idx))
    X_train = X[train_idx]
    y_train = label_array[train_idx]
    id_train = [id_array[i] for i in train_idx.tolist()]

    X_test = X[test_idx]
    y_test = label_array[test_idx]
    id_test = [id_array[i] for i in test_idx.tolist()]

    print('build ehr graph,', feature_matrix.shape)
    ehr_graph = build_ehr_nx_graph(range(f_feature_matrix.shape[0]), f_feature_matrix)  

    train_gDataset = GraphEHRDataset(ehr_graph)
    train_dataset = EHRDataset(X_train, id_train, y_train)  
    test_dataset = EHRDataset(X_test, id_test, y_test) 

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,     num_workers=1) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,     num_workers=1)

    yield fold, train_loader, test_loader, train_gDataset, None



def generate_gene_triplets(
    gene_matrix, num_id, num_gene, num_triplets, neg_trials_factor=1.5
):
    pesid = list(range(num_id))  # patient IDs
    snpid = list(range(num_id, num_id + num_gene))  # gene IDs with offset
    if not  isinstance(gene_matrix, np.ndarray):
        gene_matrix = gene_matrix.detach().cpu().numpy()
    row_idx, col_idx = np.where(gene_matrix > 0)
    pairs = np.stack([row_idx, col_idx + num_id], axis=1)  # positive (p, g)
    positive_pair_set = set(map(tuple, pairs))

    # Sample num_triplets positive anchors
    rand_idx = np.random.choice(len(pairs), size=num_triplets, replace=False)
    selected_pos_pairs = pairs[rand_idx]  # shape: [num_triplets, 2]

    triplets = []
    # trials = int(num_triplets * neg_trials_factor)

    for anchor, pos_gene in selected_pos_pairs:
        neg_sampled = False
        while not neg_sampled:
            neg_genes = np.random.choice(snpid, size=3, replace=False) #  with offset
            for neg_gene in neg_genes:
                if (anchor, neg_gene) not in positive_pair_set:
                    triplets.append((anchor, pos_gene, neg_gene))
                    neg_sampled = True
                    break

    triplets = np.array(triplets).T
    return torch.tensor(triplets, dtype=torch.long)



def generate_gene_e_link(gene_matrix, num_id, num_gene,  num_positive_samples, num_neg_samples):
    pesid = list(range(num_id))
    snpid = list(range(num_id, num_id + num_gene))
    
    row_idx, col_idx = np.where(gene_matrix > 0)
    pairs = np.stack([row_idx, col_idx + num_id], axis=1)  
    positive_pair_set = set(map(tuple, pairs))

    rand_idx = np.random.choice(len(pairs), size=num_positive_samples, replace=False)
    positive_sample_indices = pairs[rand_idx]

    num_trials = int(num_neg_samples * 1.5)
    negative_samples = []

    while len(negative_samples) < num_neg_samples:
        sampled_perids = np.random.choice(pesid, size=num_trials, replace=True)
        sampled_snpids = np.random.choice(snpid, size=num_trials, replace=True)
        sampled_pairs = list(zip(sampled_perids, sampled_snpids))

        filtered = [pair for pair in sampled_pairs if pair not in positive_pair_set]
        negative_samples.extend(filtered)
        negative_samples = list(set(negative_samples))

    negative_samples = negative_samples[:num_neg_samples]

    return (
        torch.tensor(positive_sample_indices, dtype=torch.long).T, 
        torch.tensor(negative_samples, dtype=torch.long).T
    )


def load_all_pickle_chunks(folder_path, prefix='part'):
    year = 1
    all_files = sorted(glob.glob(os.path.join(folder_path, f"{prefix}_*_{year}year.pkl")))
    df_list = [pd.read_pickle(f) for f in all_files]
    full_df = pd.concat(df_list, axis=0, ignore_index=True)
    return full_df


class EHRDataset(Dataset):
    def __init__(self, feature_values, patient_ids, label_array):
        self.patient_ids = patient_ids
        self.features = feature_values.astype("float32")
        self.labels = label_array
        
    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        return {
            "patient_id": self.patient_ids[idx],
            "ehr_input": torch.tensor(self.features[idx], dtype=torch.float32),
            "label":torch.tensor(self.labels[idx],dtype=torch.long)
        }
    

class GraphDataset():
    def __init__(self, nx_graph):

        self.nx_graph = nx_graph
#         self.edge_type_mapping = {'e-p': 0, 'e-m': 1, 'e-d': 2, 'e-me': 3, 'd-e': 2}
        self.node_type_mapping_str = {'eid': 0, 'snp': 1}
        # Convert the NetworkX graph to PyTorch Geometric format
        self.nodes, self.edges, self.node_types, self.node_mappings, self.reverse_node_mappings,\
        self.node_type_mappings = self._convert_to_pyg()
        
#         self.node_mappings = node_mappings
#         self.reverse_node_mappings = reversed_mapping
#         self.node_type_mappings = node_type_mapping

        self.data = Data(
            x=self.nodes.view(-1, 1),
            node_type=self.node_types.view(-1, 1),
            edge_index=self.edges.t().contiguous(),
#             edge_type=self.edge_types.view(-1, 1),
#             edge_attr=self.edge_values.view(-1, 1)
        )

    def _convert_to_pyg(self):
        node_mapping = {}
        nodes = []
        edges = []
        node_types = []
#         edge_values = []
#         edge_types = []
        node_type_mapping = {}
        for node in self.nx_graph.nodes(): # use the real name of nodes nomatter howmany nodes in training/testing
            node_idx = len(nodes)
            nodes.append(node_idx)
            node_mapping[node] = node_idx # node name --> node idx 

            node_type = self.node_type_mapping_str[node.split('_')[0]] # node type --> node type concept 
            node_types.append(node_type) # node type list 
            node_type_mapping[node_idx] = node_type # given node idx --> find node type concept 
 
        for edge in self.nx_graph.edges():
            src, dst = edge
            # now treat all edges as binary, no need to tell edge type or value
            edges.append((node_mapping[src], node_mapping[dst])) # source to target 
#             edge_values.append(self.nx_graph[src][dst].get('value', 0))
#             edge_type = self.edge_type_mapping[self.nx_graph[src][dst]['type']]
#             edge_types.append(edge_type)

            edges.append((node_mapping[dst], node_mapping[src])) # target to source 
#             edge_values.append(self.nx_graph[src][dst].get('value', 0))
#             edge_type = self.edge_type_mapping[self.nx_graph[src][dst]['type']]
#             edge_types.append(edge_type)

        reverse_node_mappings = {}
        for ori_node, node_idx in node_mapping.items():  
            reverse_node_mappings[node_idx] = ori_node  # from node id to node name
        return (
            torch.tensor(nodes, dtype=torch.long),
            torch.tensor(edges, dtype=torch.long),
            torch.tensor(node_types, dtype=torch.long),
#             torch.tensor(edge_types, dtype=torch.long),
#             torch.tensor(edge_values, dtype=torch.float),
            node_mapping,
            reverse_node_mappings,
            node_type_mapping)
    


class GraphEHRDataset():
    def __init__(self, nx_graph):

        self.nx_graph = nx_graph
#         self.edge_type_mapping = {'e-p': 0, 'e-m': 1, 'e-d': 2, 'e-me': 3, 'd-e': 2}
        self.node_type_mapping_str = {'eid': 0, 'ehr': 1}
        # Convert the NetworkX graph to PyTorch Geometric format
        self.nodes, self.edges, self.node_types, self.node_mappings, self.reverse_node_mappings,\
        self.node_type_mappings = self._convert_to_pyg()
        
#         self.node_mappings = node_mappings
#         self.reverse_node_mappings = reversed_mapping
#         self.node_type_mappings = node_type_mapping

        self.data = Data(
            x=self.nodes.view(-1, 1),
            node_type=self.node_types.view(-1, 1),
            edge_index=self.edges.t().contiguous(),
#             edge_type=self.edge_types.view(-1, 1),
#             edge_attr=self.edge_values.view(-1, 1)
        )

    def _convert_to_pyg(self):
        node_mapping = {}
        nodes = []
        edges = []
        node_types = []
#         edge_values = []
#         edge_types = []
        node_type_mapping = {}
        for node in self.nx_graph.nodes(): # use the real name of nodes nomatter howmany nodes in training/testing
            node_idx = len(nodes)
            nodes.append(node_idx)
            node_mapping[node] = node_idx # node name --> node idx 

            node_type = self.node_type_mapping_str[node.split('_')[0]] # node type --> node type concept 
            node_types.append(node_type) # node type list 
            node_type_mapping[node_idx] = node_type # given node idx --> find node type concept 
 
        for edge in self.nx_graph.edges():
            src, dst = edge
            # now treat all edges as binary, no need to tell edge type or value
            edges.append((node_mapping[src], node_mapping[dst])) # source to target 
#             edge_values.append(self.nx_graph[src][dst].get('value', 0))
#             edge_type = self.edge_type_mapping[self.nx_graph[src][dst]['type']]
#             edge_types.append(edge_type)

            edges.append((node_mapping[dst], node_mapping[src])) # target to source 
#             edge_values.append(self.nx_graph[src][dst].get('value', 0))
#             edge_type = self.edge_type_mapping[self.nx_graph[src][dst]['type']]
#             edge_types.append(edge_type)

        reverse_node_mappings = {}
        for ori_node, node_idx in node_mapping.items():  
            reverse_node_mappings[node_idx] = ori_node  # from node id to node name
        return (
            torch.tensor(nodes, dtype=torch.long),
            torch.tensor(edges, dtype=torch.long),
            torch.tensor(node_types, dtype=torch.long),
#             torch.tensor(edge_types, dtype=torch.long),
#             torch.tensor(edge_values, dtype=torch.float),
            node_mapping,
            reverse_node_mappings,
            node_type_mapping)



def build_ehr_nx_graph(patient_id, patient_id_snps_vector):  # train id, train id with gene, subset of both train and both having-gene ids
    nx_graph = nx.Graph()
    
    for i, pid in enumerate(patient_id):
        nx_graph.add_node(f"eid_{pid}", node_type='eid')  # Node type 'p' for patient， real id
    
    print('add node', len(patient_id))

    snp_num = patient_id_snps_vector.shape[1]
    for i in range(snp_num):
        nx_graph.add_node(f"ehr_{i}", node_type='ehr')  # Node type 'snp' for SNPs
    print('add node', snp_num)
    
    for i, pid in enumerate(patient_id):
        for j in range(snp_num):
            if patient_id_snps_vector[i, j] == 1:  # If there's a mutation
                nx_graph.add_edge(f"eid_{pid}", f"ehr_{j}", type='ee', value=1)  # Edge type 'e-p' for patient-SNP relationship
    print('add edge', len(nx_graph.edges))

    return nx_graph



# patient_id with genetic, and 
def build_nx_graph(patient_id, patient_id_snps_vector):  # train id, train id with gene, subset of both train and both having-gene ids
    nx_graph = nx.Graph()
    
    for i, pid in enumerate(patient_id):
        nx_graph.add_node(f"eid_{pid}", node_type='eid')  # Node type 'p' for patient， real id
    
    print('add node', len(patient_id))

    snp_num = patient_id_snps_vector.shape[1]
    for i in range(snp_num):
        nx_graph.add_node(f"snp_{i}", node_type='snp')  # Node type 'snp' for SNPs
    print('add node', snp_num)
    
    for i, pid in enumerate(patient_id):
        for j in range(snp_num):
            if patient_id_snps_vector[i, j] == 1:  # If there's a mutation
                nx_graph.add_edge(f"eid_{pid}", f"snp_{j}", type='es', value=1)  # Edge type 'e-p' for patient-SNP relationship
    print('add edge')

    return nx_graph


def Ginfo(G):
    # Get number of nodes and edges
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    # List nodes and edges
    print(f"Nodes: {list(G.nodes)[:3]}")
    print(f"Edges: {list(G.edges)[:3]}")

from torch_geometric.data import Data, DataLoader
import torch 

