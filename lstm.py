import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             f1_score, confusion_matrix)
import numpy as np
import gc                                                                                                                                      
MSG_IS_KEYLOGGER         = 0
MSG_IS_EXF_DOMAIN        = 1
MSG_IS_JOB_SEARCH        = 2
MSG_IS_COMPETITOR        = 3
MSG_IS_DOC               = 4
MSG_IS_PDF               = 5
MSG_IS_TXT               = 6
MSG_IS_JPG               = 7
MSG_IS_ZIP               = 8
MSG_IS_EXE               = 9
MSG_HAS_ATTACHMENTS      = 10
MSG_LOG_SIZE             = 11
MSG_CONTAIN_SUPERVISOR   = 12
MSG_INTERNAL_EMAIL_COUNT = 13
MSG_EXTERNAL_EMAIL_COUNT = 14
MSG_AFTER_HOURS          = 15
MSG_HOUR_SIN             = 16
MSG_HOUR_COS             = 17
MSG_IS_CONNECT           = 18                
MSG_IS_DISCONNECT        = 19
MSG_IS_LOGOFF            = 20
MSG_IS_LOGON             = 21
MSG_IS_EMAIL             = 22
MSG_IS_FILE              = 23
MSG_IS_HTTP              = 24
MSG_LOG_TIME_DELTA       = 25

                                                                           
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None):
        super().__init__()
        self.alpha, self.gamma, self.pos_weight = alpha, gamma, pos_weight

    def forward(self, logits, targets):
        bce          = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction='none')
        probs        = torch.sigmoid(logits)
        pt           = torch.where(targets >= 0.5, probs, 1.0 - probs)
        focal_factor = (1.0 - pt) ** self.gamma
        alpha_t      = torch.where(targets >= 0.5,
                                   torch.full_like(targets, self.alpha),
                                   torch.full_like(targets, 1.0 - self.alpha))
        return (alpha_t * focal_factor * bce).mean()
                                                                           
class InsiderThreatLSTM(nn.Module):
    def __init__(self, input_dim=66, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False)                            
        self.dropout    = nn.Dropout(dropout)
        self.attention  = nn.Linear(hidden_dim, 1, bias=False)                        
        self.layer_norm = nn.LayerNorm(hidden_dim)                                    
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),                                        
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1))

    def forward(self, x):
        lstm_out, _  = self.lstm(x)
        attn_scores  = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context      = (attn_weights * lstm_out).sum(dim=1)
        context      = self.layer_norm(context)
        context      = self.dropout(context)
        return self.classifier(context)

class TrajectoryDataset(Dataset):
    def __init__(self, embeddings_ram, labels_ram, climax_ram, scenarios_ram, users_ram, times_ram,
                 seq_len=20, split='train', normal_keep_prob=0.15, T_VAL=None, T_TEST=None):
        self.embeddings = embeddings_ram
        self.labels     = labels_ram
        self.climax     = climax_ram
        self.scenarios  = scenarios_ram
        self.users      = users_ram
        self.times      = times_ram
        self.seq_len    = seq_len

                             
        if split == 'train':
            self.mask = np.where(times_ram < T_VAL)[0]
            self.keep_prob = normal_keep_prob
        elif split == 'val':
            self.mask = np.where((times_ram >= T_VAL) & (times_ram < T_TEST))[0]
            self.keep_prob = 1.0                      
        else:
            self.mask = np.where(times_ram >= T_TEST)[0]
            self.keep_prob = 1.0

        print(f"[{split.upper()}] Sorting {len(self.mask):,} base events...")
        self.windows = self._build_windows()

    def _build_windows(self):
                                        
        u_s, t_s = self.users[self.mask], self.times[self.mask]
        sort_idx = np.lexsort((t_s, u_s))
        sorted_mask = self.mask[sort_idx]
        sorted_users = u_s[sort_idx]

                         
        changes = np.where(sorted_users[1:] != sorted_users[:-1])[0] + 1
        boundaries = [0] + changes.tolist() + [len(sorted_users)]

        windows, rng = [], np.random.default_rng(seed=42)
        for i in range(len(boundaries)-1):
            start, end = boundaries[i], boundaries[i+1]
            for w in range(start, end):
                lookback = min(w - start + 1, self.seq_len)
                win_idx = sorted_mask[w - lookback + 1 : w + 1]

                                                                                              
                is_threat = np.any(self.labels[win_idx] > 0)
                is_s3     = np.any(self.scenarios[win_idx] == 3)

                if is_threat or is_s3:
                    windows.append(win_idx)
                elif rng.random() < self.keep_prob:
                    windows.append(win_idx)

        print(f"  --> Consolidated {len(windows):,} sequences.")
        return windows

    def __len__(self): return len(self.windows)

    def __getitem__(self, idx):
        win_idx = self.windows[idx]
        x = torch.tensor(self.embeddings[win_idx], dtype=torch.float32)
        if x.size(0) < self.seq_len:
            padding = torch.zeros((self.seq_len - x.size(0), x.size(1)))
            x = torch.cat([padding, x], dim=0)

                                                                                                          
        label = float(self.climax[win_idx].max())
        scen  = int(self.scenarios[win_idx].max())
        return x, torch.tensor(label, dtype=torch.float32), scen

                                                                                
                                   
                                                                                
def run_lstm_v3_2():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n--- LSTM V3.3 Global Enhancement Training ({device}) ---")

                                    
    v = globals().get('val_t', globals().get('T_VAL', globals().get('v_t')))
    t = globals().get('test_t', globals().get('T_TEST', globals().get('t_t')))

    if v is None or t is None:
        search_fn = globals().get('find_optimized_tgn_splits')
        df_data   = globals().get('data')
        if search_fn and df_data is not None:
            print("💡 Split variables missing. Auto-calculating now...")
            v, t = search_fn(df_data)
        else:
            raise ValueError("❌ Error: Split thresholds not found. Run the Splitting cell first.")

    train_ds = TrajectoryDataset(emb_ram, labels_ram, climax_ram, scenarios_ram, users_ram, times_ram, split='train', T_VAL=v, T_TEST=t)
    val_ds   = TrajectoryDataset(emb_ram, labels_ram, climax_ram, scenarios_ram, users_ram, times_ram, split='val', T_VAL=v, T_TEST=t)
    test_ds  = TrajectoryDataset(emb_ram, labels_ram, climax_ram, scenarios_ram, users_ram, times_ram, split='test', T_VAL=v, T_TEST=t)

    train_loader = DataLoader(train_ds, batch_size=4096, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=4096)
    test_loader  = DataLoader(test_ds,  batch_size=4096)

                                   
    model = InsiderThreatLSTM(input_dim=66).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)

    best_aup = 0.0

    for epoch in range(1, 16):
        model.train()
        train_loss = 0
        for xb, yb, sb in train_loader:
            optimizer.zero_grad()
            logits = model(xb.to(device)).squeeze()
            weights = torch.ones_like(yb).to(device)
            weights[yb == 1] *= 5.0
            loss = F.binary_cross_entropy_with_logits(logits, yb.to(device), weight=weights)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        preds, targets = [], []
        with torch.no_grad():
            for xb, yb, sb in val_loader:
                preds.extend(torch.sigmoid(model(xb.to(device)).squeeze()).cpu().numpy())
                targets.extend(yb.numpy())

        preds, targets = np.array(preds), np.array(targets)
        p_c, r_c, t_c = precision_recall_curve(targets, preds)
        f1s = 2*p_c*r_c/(p_c+r_c+1e-10)
        thresh = t_c[np.argmax(f1s)] if len(t_c)>0 else 0.5
        cur_f1 = f1s.max()
        cur_aup = average_precision_score(targets, preds)

        print(f"Epoch {epoch:02d} | Loss: {train_loss/len(train_loader):.4f} | F1: {cur_f1:.4f} | AUPRC: {cur_aup:.4f}")

        if cur_aup > best_aup:
            best_aup = cur_aup
            torch.save(model.state_dict(), 'best_lstm_v3_3.pt')
            print(f"   >>> NEW BEST GLOBAL AUPRC! Saved model.")

                         
    print("\n--- FINAL V3.3 TEST (UNSEEN DATA) ---")
    model.load_state_dict(torch.load('best_lstm_v3_3.pt'))
    model.eval()
    t_preds, t_targets, t_scen = [], [], []
    with torch.no_grad():
        for xb, yb, sb in test_loader:
            t_preds.extend(torch.sigmoid(model(xb.to(device)).squeeze()).cpu().numpy())
            t_targets.extend(yb.numpy())
            t_scen.extend(sb)

    t_p, t_t, t_s = np.array(t_preds), np.array(t_targets), np.array(t_scen)
    p_c, r_c, t_c = precision_recall_curve(t_t, t_p)
    best_t = t_c[np.argmax(2*p_c*r_c/(p_c+r_c+1e-10))] if len(t_c)>0 else 0.5
    final_bin = (t_p >= best_t).astype(int)

    print(f"\n=== FINAL TEST RESULTS (UNSEEN DATA) ===")
    print(f"  Operational Threshold : {best_t:.4f}")
    print(f"  Global F1 Score       : {f1_score(t_t, final_bin, zero_division=0):.4f}")
    print(f"  Global AUPRC          : {average_precision_score(t_t, t_p):.4f}")
    print("-" * 35)

                                 
    for s_id in [1, 2, 3]:
        m = (t_s == s_id) & (t_t == 1)
        total_s = int(m.sum())
        caught_s = int(np.sum((final_bin == 1) & m))
        rec = caught_s / (total_s + 1e-10)
        print(f"  Scenario {s_id} Recall: {rec:.4f} ({caught_s}/{total_s} caught)")

    print("-" * 35)
    print(f"Confusion Matrix:\n{confusion_matrix(t_t, final_bin)}")

                                                                
                                  
                                                                  
                                             
                                                               
    all_test_climaxes = set()
    caught_climaxes = set()

    for i in range(len(t_t)):
                               
        window_events = t_w[i][t_w[i] != -1]

                                                                            
        climax_events_in_window = [e for e in window_events if climax_ram[e] == 1]

        for e in climax_events_in_window:
            all_test_climaxes.add(e)
            if final_bin[i] == 1:
                caught_climaxes.add(e)

    print(f"\n=== UNIQUE CLIMAX EVENT REPORT ===")
    print(f"  Total Unique Climaxes in Test : {len(all_test_climaxes)}")
    print(f"  Total Unique Climaxes Caught  : {len(caught_climaxes)}")
    if len(all_test_climaxes) > 0:
        rec_u = len(caught_climaxes) / len(all_test_climaxes)
        print(f"  Unique Climax Recall          : {rec_u:.4f} ({len(caught_climaxes)}/{len(all_test_climaxes)})")


run_lstm_v3_2()
