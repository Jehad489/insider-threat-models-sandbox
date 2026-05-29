import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, average_precision_score, confusion_matrix, precision_score

@torch.no_grad()
def run_standalone_evaluation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- RUNNING STANDALONE EVALUATION ({device}) ---")
    
    v = globals().get('val_t', globals().get('T_VAL', globals().get('v_t')))
    t = globals().get('test_t', globals().get('T_TEST', globals().get('t_t')))
    
    if v is None or t is None:
        print("❌ Error: Cannot find split points in RAM. You must run the data splitting cell first.")
        return

    test_ds  = TrajectoryDataset(emb_ram, labels_ram, climax_ram, scenarios_ram, users_ram, times_ram, split='test', T_VAL=v, T_TEST=t)
    test_loader  = DataLoader(test_ds,  batch_size=4096)
    
    model = InsiderThreatLSTM(input_dim=66).to(device)
    model.load_state_dict(torch.load('best_lstm_v3_3.pt'))
    model.eval()
    
    t_preds, t_targets, t_scen = [], [], []
    for xb, yb, sb in test_loader:
        t_preds.extend(torch.sigmoid(model(xb.to(device)).squeeze()).cpu().numpy())
        t_targets.extend(yb.numpy())
        t_scen.extend(sb)
        
    t_p, t_t = np.array(t_preds), np.array(t_targets)
    
    p_c, r_c, t_c = precision_recall_curve(t_t, t_p)
    best_t = t_c[np.argmax(2*p_c*r_c/(p_c+r_c+1e-10))] if len(t_c)>0 else 0.5
    final_bin = (t_p >= best_t).astype(int)
    
    print(f"\n=== GLOBAL TEST RESULTS ===")
    print(f"  Operational Threshold : {best_t:.4f}")
    print(f"  Global F1 Score       : {f1_score(t_t, final_bin, zero_division=0):.4f}")
    print(f"  Global AUPRC          : {average_precision_score(t_t, t_p):.4f}")
    
    g_precision = precision_score(t_t, final_bin, zero_division=0)
    print(f"  Global Precision      : {g_precision:.4f} (True Positives / All Flags)")
    print("-" * 35)
    print(f"Confusion Matrix:\n{confusion_matrix(t_t, final_bin)}")

                                                         
    t_w = test_ds.windows
    all_test_climaxes, caught_climaxes = set(), set()

    for i in range(len(t_t)):
        window_events = t_w[i]
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

run_standalone_evaluation()
