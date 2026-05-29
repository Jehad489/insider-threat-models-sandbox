import torch
from torch_geometric.data import TemporalData
import numpy as np
import pandas as pd

data = data.drop(columns=['id', 'log_type', 'activity', 'date', 'fractional_hour'], errors='ignore')
                                                 
unique_users = data['user'].unique()
unique_pcs   = data['pc'].unique()
num_users    = len(unique_users)
num_pcs      = len(unique_pcs)

user_to_id = {u: i for i, u in enumerate(unique_users)}
pc_to_id   = {p: i + num_users for i, p in enumerate(unique_pcs)}

data['src_idx'] = data['user'].map(user_to_id)
data['dst_idx'] = data['pc'].map(pc_to_id)

users['node_idx'] = users['user_id'].map(user_to_id)
users = users.sort_values('node_idx').reset_index(drop=True)
                                                         
data = data.sort_values(by='ts')
                                                         
data['scenario'] = data['scenario'].astype(str).replace(
    {'Normal': '0', '0': '0', '1': '1', '2': '2', '3': '3'}
).astype(int)
scenario_tensor = torch.tensor(data['scenario'].values, dtype=torch.int8)
                                                            
data['y_lazy'] = (data['scenario'] != 0).astype('float32')
y = torch.tensor(data['y_lazy'].values, dtype=torch.float32)

                                                                   
climax_tensor = torch.tensor(data['is_climax'].values, dtype=torch.int8)

                                                
for s in [1, 2, 3]:
    n = int((data['scenario'] == s).sum())
    cl = int(((data['scenario'] == s) & (data['is_climax'] == 1)).sum())
    print(f"  Scenario {s}: {n:,} window events | {cl:,} climax points")
print(f"  Total positives (lazy): {int(data['y_lazy'].sum()):,}")

                                                           
                                                    
                                                                  
                                                            
 
                                      
 
                                                                 
                                                                    
 
                
                                                       
                                                       
                                                              
                                                              
                                                             
                                                              
                                                           
                                                                    
                                        
feature_cols = [
    'is_keylogger',                                   
    'is_exf_domain',                                  
    'is_job_search_domain',                           
    'is_competitor_domain',                           
    'is_doc',                                         
    'is_pdf',                                         
    'is_txt',                                         
    'is_jpg',                                         
    'is_zip',                                         
    'is_exe',                                         
    'has_attachments',                                
    'log_size',                                              
    'contain_supervisor',                             
    'internal_emails_count',                                            
    'external_emails_count',                                            
    'after_working_hours',                            
    'hour_sin',                                        
    'hour_cos',                                        
    'is_Connect',                                                    
    'is_Disconnect',                                  
    'is_Logoff',                                      
    'is_Logon',                                       
    'is_email',                                       
    'is_file',                                        
    'is_http',                                        
    'log_time_delta',                                        
]

MSG_FEATURE_INDEX = {name: i for i, name in enumerate(feature_cols)}

                                                         
clean_msg = data[feature_cols].fillna(0).astype('float32').copy()

                                                  
LOG1P_COLS = ['log_size', 'log_time_delta', 'internal_emails_count', 'external_emails_count']
for col in LOG1P_COLS:
    clean_msg[col] = np.log1p(clean_msg[col])

                                          
                                                                                    
                                                                        
print("\nApplying Min-Max Scaling to edge features...")
for col in feature_cols:
    c_min = clean_msg[col].min()
    c_max = clean_msg[col].max()
    if c_max > c_min:
        clean_msg[col] = (clean_msg[col] - c_min) / (c_max - c_min)
    else:
        clean_msg[col] = 0.0                  

                                                                      
print("\nEdge feature post-normalization ranges:")
for col in feature_cols:
    lo = clean_msg[col].min()
    hi = clean_msg[col].max()
    tag = " ← rescale"
    print(f"  {col:<28} [{lo:7.3f}, {hi:7.3f}]{tag}")

msg = torch.tensor(clean_msg.values, dtype=torch.float32)

                                                           
                                                            
                                                              
                                                                             
                                                                      
                                                                    
                                                                   
                            
                                                           
src = torch.tensor(data['src_idx'].values, dtype=torch.long)
dst = torch.tensor(data['dst_idx'].values, dtype=torch.long)
t   = torch.tensor(data['ts'].values,      dtype=torch.long)

                                                           
                            
                                                           
temporal_data = TemporalData(
    src=src, dst=dst, t=t, msg=msg, y=y,
    scenario=scenario_tensor,
    climax=climax_tensor
)
print(f"\nGraph Instantiated! {temporal_data}")

                                                           
                                                     
                                                           
users_encoded        = users.drop(columns=['user_id', 'node_idx'], errors='ignore')
user_static_features = torch.tensor(users_encoded.values, dtype=torch.float32)
print(f"User Initial Embeddings Ready! Shape: {user_static_features.shape}")
