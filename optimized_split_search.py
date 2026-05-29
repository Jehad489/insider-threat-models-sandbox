
import numpy as np
                                                                                         
np.random.seed(42)

removal_percentage = 0.50
print(f"Original DataFrame shape: {data.shape}")
                                                        
threat_users = data.loc[data['is_threat'] == 1, 'user'].unique()
is_non_threat_user = ~data['user'].isin(threat_users)
                                                                                 
is_candidate = is_non_threat_user & data['log_type'].isin(['email', 'http'])
                                 
candidate_indices = data.index[is_candidate]
                                                     
num_to_remove = int(len(candidate_indices) * removal_percentage)
if num_to_remove > 0:
                                                                          
    indices_to_drop = np.random.choice(candidate_indices, num_to_remove, replace=False)

                                
    data = data.drop(indices_to_drop)

print(f"Dropped {num_to_remove:,} events in seconds.")
print(f"New DataFrame shape: {data.shape}")

                                  
import torch
import numpy as np

def add_climax_column_s1_only(data):
    """
    Focuses exclusively on Scenario 1 (WikiLeaks).
    Ignores Scenario 2 and 3 to maximize S1 detection purity.
    """
    print("Labeling Scenario 1 Trajectories...")
    sc_str = data['scenario'].astype(str)

                                                             
    data['is_threat_window'] = (sc_str == '1').astype(int)

                                                  
    data['is_climax'] = 0
    s1_rule = (sc_str == '1') & ((data['is_exf_domain'].astype(float) > 0) | (data['is_http'].astype(float) > 0))
    data.loc[s1_rule, 'is_climax'] = 1

    cl_count = data['is_climax'].sum()
    th_count = data['is_threat_window'].sum()
    ratio = th_count / cl_count if cl_count > 0 else 0

    print(f"  --> Identified {cl_count:,} S1 Climax Points.")
    print(f"  --> Identified {th_count:,} S1 Threat Events.")
    print(f"  --> Current Global S1 Ratio: {ratio:.1f} (Target: 5.0)")

    return data

def find_optimized_tgn_splits(data):
    """
    Finds split points that preserve FULL S1 trajectories.
    Enforces a reporting ratio of 5:1 for Scenario 1.
    """
    data = add_climax_column_s1_only(data)

                            
    threats = data[data['is_threat_window'] == 1][['ts', 'user']].sort_values(['user', 'ts'])
    sessions = []
    if len(threats) > 0:
        threats['gap'] = (threats['user'] != threats['user'].shift(1)) |\
                         (threats['ts'].diff() > 3600*24*7)
        threats['sid'] = threats['gap'].cumsum()
        sessions = threats.groupby('sid').agg(s=('ts', 'min'), e=('ts', 'max')).values.tolist()
        print(f"  --> Locked {len(sessions)} Atomic S1 Windows.")

    temp_df = data[['ts', 'scenario', 'is_climax', 'is_threat_window']].sort_values('ts').copy()
    ts, sc, cl, th = temp_df['ts'].values, temp_df['scenario'].values, temp_df['is_climax'].values, temp_df['is_threat_window'].values
    cl_idx = np.where(cl == 1)[0]

                                                            
    val_t, test_t = ts[cl_idx[int(len(cl_idx)*0.70)]], ts[cl_idx[int(len(cl_idx)*0.85)]]

    def lock(t):
        for s, e in sessions:
            if s <= t <= e: return s - 1
        return t

    val_t, test_t = lock(val_t), lock(test_t)

    def report(name, m):
        s_v, c_v, t_v = sc[m].astype(str), cl[m], th[m]
        print(f"[{name}] {len(s_v):,} total events")
                                                                           
        t1 = np.sum(t_v == 1)
        c1 = np.sum(c_v == 1)
        r1 = t1 / c1 if c1 > 0 else 0

        print(f"  SCENARIO 1 | Threats: {t1:<6} | Climaxes: {c1:<6} | Ratio: {r1:.1f}")

                                                        
        t_other = np.sum((s_v != '1') & (s_v != '0') & (s_v != 'Normal'))
        print(f"  IGNORED    | S2/S3 Events found: {t_other}")
        print("-" * 50)

    report("TRAIN", ts < val_t)
    report("VAL", (ts >= val_t) & (ts < test_t))
    report("TEST", ts >= test_t)

    return int(val_t), int(test_t)
