import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import time
import os

def main():
    t0 = time.time()
    print("======================================================================")
    print("ADDING ML PREDICTIONS TO STAR SCHEMA")
    print("======================================================================")
    
    predictions_path = 'outputs/heart_disease_with_predictions.xlsx'
    star_schema_path = 'heart_disease_star_schema_v2.xlsx'
    
    if not os.path.exists(predictions_path):
        print(f"Error: {predictions_path} not found. Please run predictions first.")
        return
        
    if not os.path.exists(star_schema_path):
        print(f"Error: {star_schema_path} not found in workspace.")
        return
        
    print("Reading ML predictions (V2 Optimized) ...")
    pred_cols = ['PatientID', 'ML_Probability', 'ML_Prediction', 'Patient_Segment', 'Patient_Segment_Label']
    pred_df = pd.read_excel(predictions_path, usecols=pred_cols)
    print(f"Loaded predictions: {pred_df.shape}  ({time.time()-t0:.1f}s)")
    
    t1 = time.time()
    print("\nReading star schema Excel file (loading all sheets) ...")
    with pd.ExcelFile(star_schema_path) as xl:
        sheets = {}
        for name in xl.sheet_names:
            t_sheet = time.time()
            print(f"  → Reading sheet: {name} ...", end="", flush=True)
            sheets[name] = pd.read_excel(xl, name)
            print(f" Done ({time.time()-t_sheet:.1f}s)")
    print(f"Loaded star schema sheets. Total read time: {time.time()-t1:.1f}s")
    
    # Merge predictions into fact_Diagnosis
    print("\nMerging predictions into fact_Diagnosis ...")
    if 'fact_Diagnosis' not in sheets:
        print("Error: fact_Diagnosis sheet not found in star schema.")
        return
        
    fact_diag = sheets['fact_Diagnosis']
    print(f"  → Original fact_Diagnosis shape: {fact_diag.shape}")
    
    # Drop target columns if they already exist to avoid duplicate suffix columns
    cols_to_drop = ['ML_Probability', 'ML_Prediction', 'Patient_Segment', 'Patient_Segment_Label']
    existing_drops = [c for c in cols_to_drop if c in fact_diag.columns]
    if existing_drops:
        print(f"  → Dropping existing columns: {existing_drops}")
        fact_diag = fact_diag.drop(columns=existing_drops)
        
    # Perform left join on PatientID
    fact_diag = pd.merge(fact_diag, pred_df, on='PatientID', how='left')
    sheets['fact_Diagnosis'] = fact_diag
    print(f"  → Updated fact_Diagnosis shape : {fact_diag.shape}")
    
    t2 = time.time()
    print("\nWriting updated sheets back to star schema Excel file ...")
    # Write to a temporary file first, then replace (for safety)
    temp_path = 'heart_disease_star_schema_v2_temp.xlsx'
    
    with pd.ExcelWriter(temp_path, engine='openpyxl') as writer:
        for name, df in sheets.items():
            t_write = time.time()
            print(f"  → Writing sheet: {name} ...", end="", flush=True)
            df.to_excel(writer, sheet_name=name, index=False)
            print(f" Done ({time.time()-t_write:.1f}s)")
            
    print("\nReplacing original star schema file with updated version ...")
    if os.path.exists(star_schema_path):
        os.remove(star_schema_path)
    os.rename(temp_path, star_schema_path)
    
    print("======================================================================")
    print(f"STAR SCHEMA SUCCESSFULLY ENRICHED IN {time.time() - t0:.1f}s")
    print("======================================================================")

if __name__ == '__main__':
    main()
