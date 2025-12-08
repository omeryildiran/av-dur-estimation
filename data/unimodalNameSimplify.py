import os
import pandas as pd
def simplify_unimodal_name(filename):
    baseName = os.path.basename(filename)
    df= pd.read_csv(baseName)
    print(f"Simplifying filename: {baseName}")
    if baseName.split('_')[1] == "auditoryDurEst":
        simpleName=baseName.split('_')[0]+'_auditory.csv'
    elif baseName.split('_')[1] == "visualDurEst":
        simpleName=baseName.split('_')[0]+'_visual.csv'

    print(f"New simplified filename: {simpleName}")

    df.to_csv(simpleName)

    

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python unimodalNameSimplify.py file1.csv file2.csv ...")
        sys.exit(1)
    
    filenames = sys.argv[1:]  # The rest are input CSVs
    for file in filenames:
        simplify_unimodal_name(file)
        #print(f"Simplified filename for {file}")
        print("\n")
        
    print("Unimodal filenames simplified successfully.")




