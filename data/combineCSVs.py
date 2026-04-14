import pandas as pd
import sys
import os

def merge_sessions(*csv_filenames):
    dfs = []
    for path in csv_filenames:
        if os.path.exists(path):
            print(f"\n\nReading: {path}")
            df_tmp=pd.read_csv(path)
            df_tmp['participantID']=os.path.basename(path).split('_')[0]
            if "VisualPSE" not in df_tmp.columns:
                if 'recordedDurVisualStandard' in df_tmp.columns and 'standardDur' in df_tmp.columns and 'conflictDur' in df_tmp.columns:
                    df_tmp["VisualPSE"]=df_tmp['recordedDurVisualStandard'] -df_tmp["standardDur"]-df_tmp['conflictDur']
                else:
                    pass

            dfs.append(df_tmp)
        else:
            print(f"Warning: File not found - {path}")
    
    if not dfs:
        raise ValueError("No valid CSV files provided.")
    
    merged_df = pd.concat(dfs, ignore_index=True)
    return merged_df

if __name__ == "__main__":
    print("\n")

    if len(sys.argv) < 3:
        print("Usage: python combineCSVs.py outputName.csv file1.csv file2.csv ...")
        sys.exit(1)
    
    outputName =sys.argv[1]# .split('_')[0][:2]+'_all.csv'#sys.argv[1]  # First argument is output filename
    
    if os.path.exists(outputName):
        print(f"Warning: Output file '{outputName}' already exists and will be removed NOW.")
        # delete the existing file
        os.remove(outputName)

    filenames = sys.argv[1:]  # The rest are input CSVs

    combined = merge_sessions(*filenames)
    combined.to_csv(outputName, index=False)
    print(f"CSV files combined successfully and saved as '{outputName}'")


"""
For single participant combinations:
python combineCSVs.py ln1_all.csv "ln1_mainExpAvDurEstimate_2025-05-14_17h31.28.988.csv" "ln1_mainExpAvDurEstimate_2025-05-12_11h14.48.428.csv"
python combineCSVs.py ln1_all.csv "ln2_mainExpAvDurEstimate_2025-06-12_13h54.19.479.csv" "ln2_mainExpAvDurEstimate_2025-06-04_11h36.41.211.csv" "ln2_mainExpAvDurEstimate_2025-06-14_16h02.00.583.csv"
"ln2_mainExpAvDurEstimate_2025-06-12_14h20.21.128.csv" "ln2_mainExpAvDurEstimate_2025-06-07_13h18.21.252.csv" "ln2_mainExpAvDurEstimate_2025-06-14_15h40.10.696.csv"

For unimodal auditory:
python combineCSVs.py all_auditory.csv "qs_auditoryDurEst_2025-06-06_15h47.33.051.csv" "as_auditoryDurEst_2025-06-20_17h24.33.242.csv" 
"mt_auditoryDurEst_2025-06-16_12h17.46.950.csv" 
"oy_auditoryDurEst_2025-04-24_14h30.44.137 copy.csv" "DT_auditoryDurEst_2025-06-10_13h06.44.870.csv"
"mh_auditoryDurEst_2025-06-09_13h11.05.687.csv" "SX_auditoryDurEst_2025-06-04_15h03.27.972.csv" "LN_auditoryDurEst_2025-06-03_16h37.04.010.csv" 
"ML_auditoryDurEst_2025-06-03_14h43.08.557.csv" "HH_auditoryDurEst_2025-05-23_11h40.57.375.csv" "IP_auditoryDurEst_2025-05-28_11h39.34.894.csv" 
"oy_auditoryDurEst_2025-03-26_22h31.24.990.csv"
"LC_auditoryDurEst_2025-05-23_16h37.43.184.csv"


For unimodal visual:

python combineCSVs.py all_visual.csv "as_visualDurEst_2025-06-20_17h48.59.604.csv" "DT_visualDurEst_2025-06-10_13h25.28.213.csv" 
"HH_visualDurEst_2025-05-23_11h59.23.202.csv" "IP_visualDurEst_2025-05-27_14h45.22.732.csv" "LN_visualDurEst_2025-06-03_16h55.40.649.csv"
"mh_visualDurEst_2025-06-09_13h30.06.112.csv" "ln_visualDurEst_2025-06-03_16h54.56.055.csv" "ML_visualDurEst_2025-06-03_14h23.28.223.csv"
"mt_visualDurEst_2025-06-16_12h38.51.056.csv" "oy_visualDurEst_2025-06-22_12h30.03.499.csv" "qs_visualDurEst_2025-06-06_16h10.47.255.csv"
"sx_visualDurEst_2025-06-04_15h25.14.957.csv" 


For main:

python combineCSVs.py all_main.csv "as_all.csv" "dt_all.csv" "HH_all.csv" "ip_all.csv" "ln1_all.csv" "ln2_all.csv" "sx_all.csv" "qs_all.csv" "oy_all.csv" "mt_all.csv" "ml_all.csv" "mh_all.csv" 
python combineCSVs.py all_woBiasedParticipants.csv "as_all.csv" "dt_all.csv" "hh_all.csv" "ln2_all.csv" "sx_all.csv" "qs_all.csv" "oy_all.csv" "mt_all.csv"  "mh_all.csv" 
python combineCSVs.py all_wo_ln1.csv "as_all.csv" "dt_all.csv" "hh_all.csv" "ip_all.csv" "ln2_all.csv" "sx_all.csv" "qs_all.csv" "oy_all.csv" "mt_all.csv" "ml_all.csv" "mh_all.csv" 

For cross-modal:

python combineCSVs.py all_crossModal.csv "DT_bimodalDurEst_2025-06-10_13h33.33.743.csv" "oy_bimodalDurEst_2025-04-17_19h32.55.390 copy.csv" "ln_bimodalDurEst_2025-04-30_11h53.06.525.csv"
"LN_bimodalDurEst_2025-06-03_17h04.55.993.csv" "as_bimodalDurEst_2025-06-20_18h00.22.264.csv" "qs_bimodalDurEst_2025-06-09_15h04.06.182.csv" 
"mt_bimodalDurEst_2025-06-16_12h50.25.740 2.csv" 
"sx_bimodalDurEst_2025-06-04_15h37.48.842.csv" "IP_bimodalDurEst_2025-05-27_15h03.01.323.csv" "ml_bimodalDurEst_2025-06-03_15h10.42.828.csv"
"HH_bimodalDurEst_2025-05-23_12h07.33.994.csv"  "mh_bimodalDurEst_2025-06-09_13h38.22.208.csv"

"""
