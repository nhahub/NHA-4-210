import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import time
import os

def fix_star_schema():
    path = 'heart_disease_star_schema_v2.xlsx'
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Skipping.")
        return
        
    t_start = time.time()
    print(f"Reading {path} ...")
    with pd.ExcelFile(path) as xl:
        sheets = {name: pd.read_excel(xl, name) for name in xl.sheet_names}
    
    print("Fixing nulls in dim_Conditions...")
    if 'dim_Conditions' in sheets:
        null_count = sheets['dim_Conditions']['Comorbidity_Level'].isna().sum()
        print(f"  → Found {null_count:,} nulls in dim_Conditions['Comorbidity_Level']")
        sheets['dim_Conditions']['Comorbidity_Level'] = sheets['dim_Conditions']['Comorbidity_Level'].fillna('None')
        print(f"  → Fixed. Remaining nulls: {sheets['dim_Conditions']['Comorbidity_Level'].isna().sum()}")
    else:
        print("  → Warning: 'dim_Conditions' sheet not found.")
    
    print(f"Writing sheets back to {path} ...")
    temp_path = 'heart_disease_star_schema_v2_temp.xlsx'
    with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
        for name, df in sheets.items():
            t_write = time.time()
            print(f"  → Writing sheet: {name} ...", end="", flush=True)
            df.to_excel(writer, sheet_name=name, index=False)
            print(f" Done ({time.time()-t_write:.1f}s)")
            
    print("Replacing original file...")
    if os.path.exists(path):
        os.remove(path)
    os.rename(temp_path, path)
    print(f"Star schema file updated successfully in {time.time()-t_start:.1f}s.\n")

def fix_project_full():
    path = 'heart_disease_project_full.xlsx'
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Skipping.")
        return
        
    t_start = time.time()
    print(f"Reading {path} ...")
    with pd.ExcelFile(path) as xl:
        sheets = {name: pd.read_excel(xl, name) for name in xl.sheet_names}
        
    print("Fixing nulls in Main_Dataset...")
    for name in sheets:
        if 'Comorbidity_Level' in sheets[name].columns:
            null_count = sheets[name]['Comorbidity_Level'].isna().sum()
            print(f"  → Found {null_count:,} nulls in {name}['Comorbidity_Level']")
            sheets[name]['Comorbidity_Level'] = sheets[name]['Comorbidity_Level'].fillna('None')
            print(f"  → Fixed. Remaining nulls: {sheets[name]['Comorbidity_Level'].isna().sum()}")
            
    print(f"Writing sheets back to {path} ...")
    temp_path = 'heart_disease_project_full_temp.xlsx'
    with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
        for name, df in sheets.items():
            t_write = time.time()
            print(f"  → Writing sheet: {name} ...", end="", flush=True)
            df.to_excel(writer, sheet_name=name, index=False)
            print(f" Done ({time.time()-t_write:.1f}s)")
            
    print("Replacing original file...")
    if os.path.exists(path):
        os.remove(path)
    os.rename(temp_path, path)
    print(f"Full project file updated successfully in {time.time()-t_start:.1f}s.\n")

def fix_predictions():
    path = 'outputs/heart_disease_with_predictions.xlsx'
    if not os.path.exists(path):
        print(f"Warning: {path} not found. Skipping.")
        return
        
    t_start = time.time()
    print(f"Reading {path} ...")
    with pd.ExcelFile(path) as xl:
        sheets = {name: pd.read_excel(xl, name) for name in xl.sheet_names}
        
    print("Fixing nulls in predictions...")
    for name in sheets:
        if 'Comorbidity_Level' in sheets[name].columns:
            null_count = sheets[name]['Comorbidity_Level'].isna().sum()
            print(f"  → Found {null_count:,} nulls in {name}['Comorbidity_Level']")
            sheets[name]['Comorbidity_Level'] = sheets[name]['Comorbidity_Level'].fillna('None')
            print(f"  → Fixed. Remaining nulls: {sheets[name]['Comorbidity_Level'].isna().sum()}")
            
    print(f"Writing sheets back to {path} ...")
    temp_path = 'heart_disease_with_predictions_temp.xlsx'
    with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
        for name, df in sheets.items():
            t_write = time.time()
            print(f"  → Writing sheet: {name} ...", end="", flush=True)
            df.to_excel(writer, sheet_name=name, index=False)
            print(f" Done ({time.time()-t_write:.1f}s)")
            
    print("Replacing original file...")
    if os.path.exists(path):
        os.remove(path)
    os.rename(temp_path, path)
    print(f"Predictions file updated successfully in {time.time()-t_start:.1f}s.\n")

if __name__ == '__main__':
    t0 = time.time()
    print("======================================================================")
    print("STARTING COMORBIDITY NULLS FIX")
    print("======================================================================")
    fix_star_schema()
    fix_project_full()
    fix_predictions()
    print("======================================================================")
    print(f"ALL FILES UPDATED IN {time.time()-t0:.1f}s")
    print("======================================================================")
