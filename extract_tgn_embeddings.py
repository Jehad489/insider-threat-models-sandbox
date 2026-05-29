import torch
import torch.nn.functional as F
import numpy as np
import os
from torch_geometric.loader import TemporalDataLoader
from tqdm import tqdm

                        
if 'model' not in globals() or 'temporal_data' not in globals():
    raise NameError(
        "\n[MISSING OBJECTS] Your memory was wiped on disconnect.\n"
        "Please run Tensorization and TGN cells (V3.0) first!"
    )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Extraction Using V3.0 Cognitive Engine on: {device}")


def extract_and_save_embeddings(model, dataloader, node_static_features):
    """
    Extracts V3.1 Cognitive Embeddings (64-dim) + Anomaly (1) + Prob (1).
    Total LSTM Input: 66 features per event.
    """
    model.eval()
    model.memory.reset_state()
    device = next(model.parameters()).device
    node_static_features = node_static_features.to(device)

    try:
        num_edges = dataloader.dataset.src.size(0)
    except Exception:
        num_edges = len(temporal_data.src)

                                                   
    DYNAMIC_DIM = 64
    TOTAL_DIM   = DYNAMIC_DIM + 2                

    print(f"\nAllocating V3.1 RAM Context...")
    print(f"  emb_ram : [{num_edges:,}, {TOTAL_DIM}] float32  ← [64 GNN + 1 Anom + 1 Prob]")
    print(f"  labels  : [{num_edges:,}] int8                 ← Threat and Climax")

    global emb_ram, msg_ram, labels_ram, climax_ram, users_ram, times_ram, scenarios_ram

    emb_ram     = np.zeros((num_edges, TOTAL_DIM), dtype=np.float32)
    labels_ram  = np.zeros((num_edges,),           dtype=np.int8)
    climax_ram  = np.zeros((num_edges,),           dtype=np.int8)
    users_ram   = np.zeros((num_edges,),           dtype=np.int64)
    times_ram   = np.zeros((num_edges,),           dtype=np.int64)           
    scenarios_ram = np.zeros((num_edges,),         dtype=np.int8)

    current_idx = 0
    print("\n--- Extracting V3.1 Behavioral Embeddings ---")

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch = batch.to(device)
            model.memory.detach()

            src, dst, t, msg_feat = batch.src, batch.dst, batch.t, batch.msg
            edge_index = torch.stack([src, dst], dim=0)

                                    
            static_src = node_static_features[src]
            static_dst = node_static_features[dst]

                                                                  
            n_id           = torch.cat([src, dst]).unique()
            z, last_update = model.memory(n_id)
            assoc          = torch.empty(model.memory.num_nodes, dtype=torch.long, device=z.device)
            assoc[n_id]    = torch.arange(n_id.size(0), device=z.device)

            z_gnn          = model.gnn(z, last_update, assoc[edge_index], t, msg_feat, static_src, static_dst)

                                    
            logits, interaction_emb = model.link_pred(
                z_gnn[assoc[src]], z_gnn[assoc[dst]], static_src, static_dst, msg_feat
            )
            prob = torch.sigmoid(logits.squeeze())

                                                    
            recon_msg = model.recon_head(
                z_gnn[assoc[src]], z_gnn[assoc[dst]], static_src, static_dst
            )
            mse_anomaly = F.mse_loss(recon_msg, msg_feat, reduction='none').mean(dim=-1)

            batch_size = src.size(0)

                                                                            
            combined = torch.cat([
                interaction_emb,
                mse_anomaly.unsqueeze(-1),
                prob.unsqueeze(-1)
            ], dim=-1)

            emb_ram[current_idx:current_idx + batch_size]    = combined.cpu().numpy()
            labels_ram[current_idx:current_idx + batch_size] = batch.y.cpu().numpy().astype(np.int8)
            climax_ram[current_idx:current_idx + batch_size] = batch.climax.cpu().numpy().astype(np.int8)
            users_ram[current_idx:current_idx + batch_size]  = src.cpu().numpy()
            times_ram[current_idx:current_idx + batch_size]  = t.cpu().numpy()           
            scenarios_ram[current_idx:current_idx + batch_size] = batch.scenario.cpu().numpy().astype(np.int8)

            current_idx += batch_size
            model.memory.update_state(src, dst, t, msg_feat)

    print(f"\nV3.1 Extraction Complete! Final Shape: {emb_ram.shape}")

    print(f"  Final Shape: {emb_ram.shape} (Ready for BiLSTM Stage)")
                                                                             
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if 'model' in globals() and 'n_static' in globals():
    print(f"--- Starting Unified Extraction Trace ({device}) ---")
    full_loader = TemporalDataLoader(temporal_data, batch_size=8192)
    extract_and_save_embeddings(model, full_loader, n_static)
else:
    print("❌ Error: 'model' or 'n_static' missing from RAM. Run the TGN cell first.")
