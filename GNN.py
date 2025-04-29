import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score


def load_data():
    try:

        print("Loading datasets...")
        drug_response = pd.read_csv('Drug_Response.csv')
        cell_line = pd.read_csv('Cell_Line_Number.csv')
        drug_target = pd.read_csv('Drug_Target.csv')
        
        print(f"Loaded data: \n"
              f"- Drug response data: {drug_response.shape} \n"
              f"- Cell line data: {cell_line.shape} \n"
              f"- Drug target data: {drug_target.shape}")
        
        drug_ids = drug_response['Drug ID'].unique()[:3]  
        cell_lines = drug_response['Cell Line Name'].unique()[:5] 
        
        drug_response_subset = drug_response[
            (drug_response['Drug ID'].isin(drug_ids)) & 
            (drug_response['Cell Line Name'].isin(cell_lines))
        ]
        
        drug_target_subset = drug_target[drug_target['Drug ID'].isin(drug_ids)]
        
        cosmic_ids = drug_response_subset['Cosmic ID'].unique()
        cell_line_subset = cell_line[cell_line['COSMIC ID'].isin(cosmic_ids)]
        
        if cell_line_subset.empty:
            cell_line_subset = cell_line.head(5)
            
        print(f"Created subsets: \n"
              f"- Drug response subset: {drug_response_subset.shape} \n"
              f"- Cell line subset: {cell_line_subset.shape} \n"
              f"- Drug target subset: {drug_target_subset.shape}")
              
        return drug_response_subset, cell_line_subset, drug_target_subset
        
    except Exception as e:
        print(f"Error loading data: {e}")

        print("Creating minimal test data...")
        
        drug_response_minimal = pd.DataFrame({
            'Drug Name': ['Camptothecin', 'Camptothecin', 'Camptothecin', 'Camptothecin'],
            'Drug ID': [1003, 1003, 1003, 1003],
            'Cell Line Name': ['PFSK-1', 'A673', 'ES5', 'ES7'],
            'Cosmic ID': [683667, 684052, 684057, 684059],
            'TCGA Classification': ['MB', 'UNCLASSIFIED', 'UNCLASSIFIED', 'UNCLASSIFIED'],
            'Tissue': ['nervous_system', 'soft_tissue', 'bone', 'bone'],
            'Tissue Sub-type': ['medulloblastoma', 'rhabdomyosarcoma', 'ewings_sarcoma', 'ewings_sarcoma'],
            'IC50': [-1.4638, -4.8695, -3.3605, -5.0494],
            'AUC': [0.9302, 0.6149, 0.7910, 0.5926],
            'Z score': [0.4331, -1.4211, -0.5995, -1.5166]
        })
        
        cell_line_minimal = pd.DataFrame({
            'Cell Line Name': ['BONNA-12', 'BONNA-12', 'BONNA-12', 'BONNA-12', 'BONNA-12'],
            'COSMIC ID': [906695, 906695, 906695, 906695, 906695],
            'GDSC Desc1': ['blood', 'blood', 'blood', 'blood', 'blood'],
            'GDSC Desc2': ['hairy_cell_leukaemia', 'hairy_cell_leukaemia', 'hairy_cell_leukaemia', 'hairy_cell_leukaemia', 'hairy_cell_leukaemia'],
            'Genetic Feature': ['cnaPANCAN1', 'cnaPANCAN2', 'cnaPANCAN3', 'cnaPANCAN4', 'cnaPANCAN5'],
            'IS Mutated': [0, 0, 0, 0, 0],
            'Recurrent Gain Loss': ['gain', 'loss', 'loss', 'loss', 'loss'],
        })
        
        drug_target_minimal = pd.DataFrame({
            'Drug name': ['Camptothecin', 'Camptothecin', 'Camptothecin', 'Camptothecin', 'Camptothecin', 'Camptothecin'],
            'Drug ID': [1003, 1003, 1003, 1003, 1003, 1003],
            'Drug target': ['TOP 1.00', 'TOP 1.00', 'TOP 1.00', 'TOP 1.00', 'TOP 1.00', 'TOP 1.00'],
            'Target Pathway': ['DNA replication', 'DNA replication', 'DNA replication', 'DNA replication', 'DNA replication', 'DNA replication'],
            'Feature Name': ['ABC91_mut', 'ABL2_mut', 'ACACA_mut', 'ACRV1B_mut', 'ACVR2A_mut', 'AFF4_mut'],
            'ic50_effect_size': [0.5598, 0.3732, 0.1592, 0.0849, 0.0838, 0.0268],
        })
        
        return drug_response_minimal, cell_line_minimal, drug_target_minimal

class GraphDataPreprocessor:
    def __init__(self, drug_response, cell_line, drug_target):
        self.drug_response = drug_response
        self.cell_line = cell_line
        self.drug_target = drug_target
        
    def preprocess(self):
        print("Preprocessing data...")

        unique_drugs = self.drug_response['Drug ID'].unique()
        unique_cell_lines = self.drug_response['Cell Line Name'].unique()
        
        drug_features = self._create_drug_features()
        cell_line_features = self._create_cell_line_features()
        
        edge_index, edge_attr, edge_mapping = self._create_edges()
        
        y = self.drug_response['IC50'].values.astype(np.float32)
        
        drug_mapping = {drug_id: idx for idx, drug_id in enumerate(unique_drugs)}
        cell_mapping = {cell_line: idx + len(unique_drugs) 
                        for idx, cell_line in enumerate(unique_cell_lines)}
        
        x = torch.cat([
            torch.FloatTensor(drug_features),
            torch.FloatTensor(cell_line_features)
        ], dim=0)
        
        edge_index = torch.LongTensor(edge_index)
        edge_attr = torch.FloatTensor(edge_attr)
        y = torch.FloatTensor(y)
        
        print(f"Processed data: \n"
              f"- Node features shape: {x.shape} \n"
              f"- Edge index shape: {edge_index.shape} \n"
              f"- Edge attr shape: {edge_attr.shape} \n"
              f"- Target shape: {y.shape}")
        
        return x, edge_index, edge_attr, y, drug_mapping, cell_mapping, edge_mapping
    
    def _create_drug_features(self):
        drug_ids = self.drug_response['Drug ID'].unique()
        
        features = []
        for drug_id in drug_ids:
            feature_vector = [drug_id, hash(str(drug_id)) % 10]
            features.append(feature_vector)
        
        return np.array(features, dtype=np.float32)
    
    def _create_cell_line_features(self):
        cell_lines = self.drug_response['Cell Line Name'].unique()
        
        features = []
        for cell_line in cell_lines:
            feature_vector = [hash(cell_line) % 100, hash(cell_line[::-1]) % 10]
            features.append(feature_vector)
        
        return np.array(features, dtype=np.float32)
    
    def _create_edges(self):
        edges = []
        edge_features = []
        edge_mapping = {} 
        
        drug_ids = self.drug_response['Drug ID'].unique()
        drug_idx_map = {drug_id: idx for idx, drug_id in enumerate(drug_ids)}
        
        cell_lines = self.drug_response['Cell Line Name'].unique()
        cell_idx_map = {cell: idx + len(drug_ids) for idx, cell in enumerate(cell_lines)}
        
        for i, row in self.drug_response.iterrows():
            drug_id = row['Drug ID']
            cell_line = row['Cell Line Name']
            
            drug_idx = drug_idx_map[drug_id]
            cell_idx = cell_idx_map[cell_line]
            
            edges.append([drug_idx, cell_idx])
            edges.append([cell_idx, drug_idx])
            
            if 'AUC' in row:
                edge_feat = [float(row['AUC'])]
            else:
                edge_feat = [0.5] 
                
            edge_features.append(edge_feat)
            edge_features.append(edge_feat)  
            
            edge_mapping[(drug_id, cell_line)] = i
        
        edge_index = np.array(edges).T  
        edge_attr = np.array(edge_features)
        
        return edge_index, edge_attr, edge_mapping


class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=1):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, edge_attr=None):

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.1, training=self.training)
        

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        

        x = self.lin(x)
        
        return x


def create_graph_data(x, edge_index, edge_attr, y, drug_mapping, cell_mapping, edge_mapping, drug_response):
    """Creates PyTorch Geometric Data objects for training"""
    
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        drug_mapping=drug_mapping,
        cell_mapping=cell_mapping,
        edge_mapping=edge_mapping
    )
    
    return data


def train(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x, data.edge_index, data.edge_attr)
    
    drug_nodes = list(range(len(data.drug_mapping)))
    drug_predictions = out[drug_nodes].squeeze()
    
    loss = criterion(drug_predictions, data.y)
    
    loss.backward()
    optimizer.step()
    
    return loss.item()


def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index, data.edge_attr)
        
        drug_nodes = list(range(len(data.drug_mapping)))
        predictions = out[drug_nodes].squeeze()
    
    return predictions

def main():
    print("Starting GNN for Personalized Drug Treatment Prediction")
    
    drug_response, cell_line, drug_target = load_data()
    
    preprocessor = GraphDataPreprocessor(drug_response, cell_line, drug_target)
    x, edge_index, edge_attr, y, drug_mapping, cell_mapping, edge_mapping = preprocessor.preprocess()
    
 
    graph_data = create_graph_data(x, edge_index, edge_attr, y, 
                                  drug_mapping, cell_mapping, edge_mapping, drug_response)
    

    in_channels = x.shape[1]
    hidden_channels = 64
    model = GNN(in_channels, hidden_channels)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    

    print("Starting training...")
    n_epochs = 100
    train_losses = []
    
    for epoch in range(n_epochs):
        loss = train(model, graph_data, optimizer, criterion)
        train_losses.append(loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Loss: {loss:.4f}")
    

    print("Evaluating model...")
    pred = evaluate(model, graph_data)
    
  
    if len(pred.shape) == 0:  
        pred = pred.reshape(1)
    

    true = graph_data.y
    
 
    if len(pred) > 1 and len(true) > 1:
        pred_np = pred.numpy()
        true_np = true.numpy()
        

        mse = mean_squared_error(true_np, pred_np)
        
     
        if np.var(true_np) > 0:
            r2 = r2_score(true_np, pred_np)
            print(f"Test R²: {r2:.4f}")
        else:
            print("Cannot calculate R² (insufficient variance in data)")
            
        print(f"Test MSE: {mse:.4f}")
    else:
        print("Not enough samples to calculate metrics. Displaying predictions vs actual:")
        for i in range(len(pred)):
            print(f"True: {true[i].item():.4f}, Predicted: {pred[i].item():.4f}")
    
 
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    print("Training loss plot saved as 'training_loss.png'")
    

    print("\nPredictions for all drug-cell line pairs:")
    

    drug_ids = list(drug_mapping.keys())
    cell_lines = list(cell_mapping.keys())
    

    predictions_table = []
    
    for drug_id in drug_ids:
        drug_idx = drug_mapping[drug_id]
        drug_name = drug_response[drug_response['Drug ID'] == drug_id]['Drug Name'].iloc[0] if 'Drug Name' in drug_response.columns else f"Drug {drug_id}"
        
        drug_predictions = []
        for cell_line in cell_lines:
            cell_idx = cell_mapping[cell_line]
            

            drug_node_pred = pred[drug_idx].item()
            
     
            if (drug_id, cell_line) in edge_mapping:
                response_idx = edge_mapping[(drug_id, cell_line)]
                actual_ic50 = y[response_idx].item()
                drug_predictions.append((cell_line, drug_node_pred, actual_ic50))
            else:
                drug_predictions.append((cell_line, drug_node_pred, None))
        
        predictions_table.append((drug_name, drug_id, drug_predictions))
    

    for drug_name, drug_id, cell_predictions in predictions_table:
        print(f"\nDrug: {drug_name} (ID: {drug_id})")
        print("-" * 60)
        print(f"{'Cell Line':<20} {'Predicted IC50':<20} {'Actual IC50':<20}")
        print("-" * 60)
        
        for cell_line, pred_ic50, actual_ic50 in cell_predictions:
            actual_str = f"{actual_ic50:.4f}" if actual_ic50 is not None else "N/A"
            print(f"{cell_line:<20} {pred_ic50:.4f}{' ':16} {actual_str:<20}")
    
 
    torch.save(model.state_dict(), 'gnn_drug_response_model.pt')
    print("\nModel saved as 'gnn_drug_response_model.pt'")
    
    print("Done!")

if __name__ == "__main__":
    main()