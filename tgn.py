import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator
from sklearn.metrics import average_precision_score, f1_score, confusion_matrix, precision_recall_curve
import numpy as np

                                              
class TimeEncoder(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.lin = Linear(1, out_channels)
    def forward(self, t):
        return torch.cos(self.lin(t.view(-1, 1)))

class GraphAttentionEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_dim, static_dim):
        super().__init__()
        self.static_proj = Linear(static_dim, 8)
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2, dropout=0.1, edge_dim=msg_dim + time_dim + 16)
        self.time_encoder = TimeEncoder(time_dim)

    def forward(self, x, last_update, edge_index, t, msg, s_src, s_dst):
        rel_t_norm = torch.log1p((last_update[edge_index[0]] - t).float().abs() / 86400.0)
        rel_t_enc  = self.time_encoder(rel_t_norm.to(x.dtype))
        s_src_p, s_dst_p = F.relu(self.static_proj(s_src)), F.relu(self.static_proj(s_dst))
        edge_attr  = torch.cat([rel_t_enc, msg, s_src_p, s_dst_p], dim=-1)
        return self.conv(x, edge_index, edge_attr)

                                             
class LinkPredictor(nn.Module):
    def __init__(self, dynamic_dim, static_dim, msg_dim):
        super().__init__()
        in_channels = (dynamic_dim * 2) + (static_dim * 2) + msg_dim
        self.lin1, self.lin2 = Linear(in_channels, 64), Linear(64, 1)
    def forward(self, z_src, z_dst, s_src, s_dst, msg):
        x = torch.cat([z_src, z_dst, s_src, s_dst, msg], dim=-1)
        interaction_emb = F.relu(self.lin1(x))
        return self.lin2(interaction_emb), interaction_emb

class ReconstructionHead(nn.Module):
    def __init__(self, dynamic_dim, static_dim, out_dim):
        super().__init__()
        in_channels = (dynamic_dim * 2) + (static_dim * 2)
        self.net = nn.Sequential(Linear(in_channels, 64), nn.ReLU(), Linear(64, out_dim))
    def forward(self, z_src, z_dst, s_src, s_dst):
        return self.net(torch.cat([z_src, z_dst, s_src, s_dst], dim=-1))

class InsiderThreatTGN(nn.Module):
    def __init__(self, num_nodes, memory_dim, time_dim, msg_dim, static_dim):
        super().__init__()
        self.memory = TGNMemory(num_nodes, msg_dim, memory_dim, time_dim, IdentityMessage(msg_dim, memory_dim, time_dim), LastAggregator())
        self.gnn = GraphAttentionEmbedding(memory_dim, memory_dim, msg_dim, time_dim, static_dim)
        self.link_pred = LinkPredictor(memory_dim, static_dim, msg_dim)
        self.recon_head = ReconstructionHead(memory_dim, static_dim, msg_dim)

    def forward(self, src, dst, t, msg, edge_index, static_src, static_dst):
        n_id = torch.cat([src, dst]).unique()
        z, last_update = self.memory(n_id)
        assoc = torch.empty(self.memory.num_nodes, dtype=torch.long, device=z.device)
        assoc[n_id] = torch.arange(n_id.size(0), device=z.device)
        z_gnn = self.gnn(z, last_update, assoc[edge_index], t, msg, static_src, static_dst)
        logit, _ = self.link_pred(z_gnn[assoc[src]], z_gnn[assoc[dst]], static_src, static_dst, msg)
        recon_msg = self.recon_head(z_gnn[assoc[src]], z_gnn[assoc[dst]], static_src, static_dst)
        self.memory.update_state(src, dst, t, msg)
        return logit, recon_msg

                      
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, logits, targets, importance_basis=None):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        probs = torch.sigmoid(logits)
        pt = torch.where(targets >= 0.5, probs, 1.0 - probs)
        alpha_t = torch.where(targets >= 0.5, torch.tensor(self.alpha, device=logits.device), torch.tensor(1.0-self.alpha, device=logits.device))
        loss = alpha_t * ((1.0 - pt) ** self.gamma) * bce
        if importance_basis is not None: loss *= importance_basis
        return loss.mean()

@torch.no_grad()
def evaluate_tgn(model, dataloader, node_static_features):
    model.eval()
    preds, targets, climaxes, scenarios = [], [], [], []
    device = next(model.parameters()).device
    for batch in dataloader:
        batch = batch.to(device)
        static_src, static_dst = node_static_features[batch.src], node_static_features[batch.dst]
        logits, _ = model(batch.src, batch.dst, batch.t, batch.msg, torch.stack([batch.src, batch.dst], dim=0), static_src, static_dst)
        preds.append(torch.sigmoid(logits.squeeze()).cpu().numpy())
        targets.append(batch.y.cpu().numpy())
        climaxes.append(batch.climax.cpu().numpy())
        scenarios.append(batch.scenario.cpu().numpy())

    preds, targets, climaxes, sc = np.concatenate(preds), np.concatenate(targets), np.concatenate(climaxes), np.concatenate(scenarios)
    ap_t = average_precision_score(targets, preds)
    p, r, t_curves = precision_recall_curve(targets, preds)
    thresh = float(t_curves[np.argmax(2*r*p/(r+p+1e-10))]) if len(t_curves)>0 else 0.5
    bin_p = (preds > thresh).astype(int)

                         
    s_rec = {}
    for s_id in [1, 2, 3]:
        m = (sc == s_id) & (targets == 1)
        s_rec[s_id] = np.sum((bin_p == 1) & m) / (m.sum() + 1e-10)

    return {
        'AUPRC_Threat': ap_t, 'F1_Threat': f1_score(targets, bin_p, zero_division=0),
        'Recall_Threat': np.sum((bin_p==1)&(targets==1))/(np.sum(targets==1)+1e-10),
        'AUPRC_Climax': average_precision_score(climaxes, preds),
        'Climax_Recall': np.sum((bin_p==1)&(climaxes==1))/(np.sum(climaxes==1)+1e-10),
        'S1_Rec': s_rec[1], 'S2_Rec': s_rec[2], 'S3_Rec': s_rec[3],
        'Threshold': thresh
    }

def train_tgn_epoch(model, loader, opt, crit, static_features, accumulation_steps=4):
    model.train(); model.memory.reset_state()
    device = next(model.parameters()).device
    total_l, total_r, total_e = 0, 0, 0
    opt.zero_grad()
    for step, batch in enumerate(loader):
        batch = batch.to(device); model.memory.detach()
        s_src, s_dst = static_features[batch.src], static_features[batch.dst]
        logits, r_msg = model(batch.src, batch.dst, batch.t, batch.msg, torch.stack([batch.src, batch.dst], dim=0), s_src, s_dst)
        imp = 1.0 + (batch.climax.float() * 4.0)
        c_loss = crit(logits.squeeze(), batch.y.float(), importance_basis=imp)
        r_loss = F.mse_loss(r_msg, batch.msg)
        ((c_loss + 0.1*r_loss)/accumulation_steps).backward()
        if (step+1)%accumulation_steps==0: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); opt.zero_grad()
        total_l += c_loss.item()*batch.num_events; total_r += r_loss.item()*batch.num_events; total_e += batch.num_events
    return total_l/total_e, total_r/total_e

                                                                                
                                                                
                                                                                
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                        
T_VAL  = globals().get('T_VAL', globals().get('val_t'))
T_TEST = globals().get('T_TEST', globals().get('test_t'))

if T_VAL is None or T_TEST is None:
    search_fn = globals().get('find_optimized_tgn_splits')
    df_data   = globals().get('data')
    if search_fn and df_data is not None:
        print("💡 Split variables missing from RAM. Auto-calculating now...")
        T_VAL, T_TEST = search_fn(df_data)
    else:
        print("⚠️ Warning: Could not auto-detect splits. Please run the Splitting cell first.")

if T_VAL and T_TEST:
    print(f"--- TGN V3.6 Master Session Starting ({device}) ---")

                           
    train_loader = TemporalDataLoader(temporal_data[temporal_data.t < T_VAL], batch_size=4096)
    val_loader   = TemporalDataLoader(temporal_data[(temporal_data.t >= T_VAL) & (temporal_data.t < T_TEST)], batch_size=4096)
    test_loader  = TemporalDataLoader(temporal_data[temporal_data.t >= T_TEST], batch_size=4096)

                                                                  
    static_dim = user_static_features.size(1)
    n_static = torch.zeros((num_users + num_pcs, static_dim))
    n_static[:num_users] = user_static_features
    n_static = n_static.to(device)

    model = InsiderThreatTGN(num_users + num_pcs, 128, 64, 26, static_dim).to(device)
    opt, crit = torch.optim.Adam(model.parameters(), lr=0.0001), FocalLoss()

                 
    best_aup = 0.0
    for epoch in range(1, 16):
        l, r = train_tgn_epoch(model, train_loader, opt, crit, n_static)
        res = evaluate_tgn(model, val_loader, n_static)
        print(f"Epoch {epoch:02d} | Loss: {l:.4f} | [S1 RECALL]: {res['S1_Rec']:.4f} | AUPRC: {res['AUPRC_Threat']:.4f}")

        if res['AUPRC_Threat'] > best_aup:
            best_aup = res['AUPRC_Threat']
            torch.save(model.state_dict(), 'tgn_v3_cognitive.pt')

                                        
    print("\n--- TGN FINAL BLIND TEST Evaluation ---")
    model.load_state_dict(torch.load('tgn_v3_cognitive.pt'))                       
    test_res = evaluate_tgn(model, test_loader, n_static)

    from sklearn.metrics import confusion_matrix
    print(f"  Test S1 Recall: {test_res['S1_Rec']:.4f}")
    print(f"  Test AUPRC    : {test_res['AUPRC_Threat']:.4f}")
    print(f"  Test F1 Score : {test_res['F1_Threat']:.4f}")
    print("-" * 35)

                                         
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for b in test_loader:
            b = b.to(device)
            l, _ = model(b.src, b.dst, b.t, b.msg, torch.stack([b.src, b.dst], dim=0), n_static[b.src], n_static[b.dst])
            all_p.extend((torch.sigmoid(l.squeeze()) > test_res['Threshold']).cpu().numpy())
            all_y.extend(b.y.cpu().numpy())

    print("Test Confusion Matrix (S1 Specialist):")
    print(confusion_matrix(all_y, all_p))

else:
    print("❌ Fatal Error: Could not establish training splits.")
                                                                                
                                               
                                                                                
@torch.no_grad()
def evaluate_tgn_full(model, dataloader, node_static_features):
    model.eval()
    preds, targets, climaxes, scenarios = [], [], [], []
    device = next(model.parameters()).device
    for batch in dataloader:
        batch = batch.to(device)
        static_src, static_dst = node_static_features[batch.src], node_static_features[batch.dst]
        logits, _ = model(batch.src, batch.dst, batch.t, batch.msg, torch.stack([batch.src, batch.dst], dim=0), static_src, static_dst)
        preds.append(torch.sigmoid(logits.squeeze()).cpu().numpy())
        targets.append(batch.y.cpu().numpy())
        climaxes.append(batch.climax.cpu().numpy())
        scenarios.append(batch.scenario.cpu().numpy())

    preds, targets, climaxes, sc = np.concatenate(preds), np.concatenate(targets), np.concatenate(climaxes), np.concatenate(scenarios)
    ap_t = average_precision_score(targets, preds)
    p, r, t_curves = precision_recall_curve(targets, preds)
    thresh = float(t_curves[np.argmax(2*r*p/(r+p+1e-10))]) if len(t_curves)>0 else 0.5
    bin_p = (preds > thresh).astype(int)

                         
    s_rec = {}
    for s_id in [1, 2, 3]:
        m = (sc == s_id) & (targets == 1)
        s_rec[s_id] = np.sum((bin_p == 1) & m) / (m.sum() + 1e-10)

    return {
        'AUPRC_Threat': ap_t, 'F1_Threat': f1_score(targets, bin_p, zero_division=0),
        'Recall_Threat': np.sum((bin_p==1)&(targets==1))/(np.sum(targets==1)+1e-10),
        'AUPRC_Climax': average_precision_score(climaxes, preds),
        'Climax_Recall': np.sum((bin_p==1)&(climaxes==1))/(np.sum(climaxes==1)+1e-10),
        'S1_Rec': s_rec[1], 'S2_Rec': s_rec[2], 'S3_Rec': s_rec[3],
        'Threshold': thresh
    }

                 
v = globals().get('val_t', globals().get('T_VAL', globals().get('v_t')))
t = globals().get('test_t', globals().get('T_TEST', globals().get('t_t')))
data_test = temporal_data[temporal_data.t >= t]
test_loader = TemporalDataLoader(data_test, batch_size=4096)

                   
test_res = evaluate_tgn_full(model, test_loader, n_static)

print("\n=== FINAL TEST RESULTS (UNSEEN DATA) ===")
print(f"  THREAT AUPRC  : {test_res['AUPRC_Threat']:.4f}")
print(f"  THREAT F1     : {test_res['F1_Threat']:.4f}")
print(f"  THREAT RECALL : {test_res['Recall_Threat']:.4f}")
print("-" * 30)
print(f"  CLIMAX RECALL : {test_res['Climax_Recall']:.4f}")
print("-" * 30)
print(f"  S1 RECALL     : {test_res['S1_Rec']:.4f}")
print(f"  S2 RECALL     : {test_res['S2_Rec']:.4f}")
print(f"  S3 RECALL     : {test_res['S3_Rec']:.4f}")
print(f"\n  Operational Threshold used: {test_res['Threshold']:.4f}")
