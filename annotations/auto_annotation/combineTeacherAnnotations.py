import json
import pandas as pd
import argparse

def read_json_to_df(file_path):
    # Read JSON lines
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
        # Use json_normalize to flatten nested dictionaries
        df = pd.json_normalize(data)
        # Rename columns to replace '.' with '_' for easier access
        df.columns = df.columns.str.replace('.', '_', regex=False)
        
        return df



def process_is_safe_columns(row):
    is_safe_columns = [col for col in row.index if col.endswith('_is_safe')]
    
    if len(is_safe_columns) == 2:
        # If there are exactly two columns, check if they're equal
        if row[is_safe_columns[0]] == row[is_safe_columns[1]]:
            return row[is_safe_columns[0]]
        else:
            return "disagree"
    elif len(is_safe_columns) > 2:
        # If there are more than two columns, use majority voting
        votes = row[is_safe_columns].value_counts()
        return votes.index[0] if votes.iloc[0] > len(is_safe_columns) // 2 else None
    else:
        # If there are fewer than two columns, return None
        return "toofewcols"





def process_json_files(file1_path, file2_path, output_file):
    # Read JSON files into pandas DataFrames
    df1 = read_json_to_df(file1_path)
    df2 = read_json_to_df(file2_path)
    
    # # Combine DataFrames based on UID
    combined_df = pd.concat([df1, df2]).groupby('UID', as_index=False).first()
    

    # # Apply the function to create a new column
    combined_df['Is_Safe_Result'] = combined_df.apply(process_is_safe_columns, axis=1)
    combined_df['Annotations_Agree'] = combined_df['Is_Safe_Result'].apply(lambda x: x not in ["disagree", "toofewcols"])

    # Create DataFrames for agreements and disagreements
    agreements_df = combined_df[combined_df['Annotations_Agree'] == True].copy()
    disagreements_df = combined_df[combined_df['Annotations_Agree'] == False].copy()


    combined_df.to_csv(output_file, index=False)
    agreements_df.to_csv("agreements_" + output_file, index=False)
    disagreements_df.to_csv("disagreements_" + output_file, index=False)


    

def main():
    parser = argparse.ArgumentParser(description="Process JSON files and compare annotations.")
    parser.add_argument("--file1",default="", help="Path to the first TeacherAnnotation JSON file")
    parser.add_argument("--file2", default="",help="Path to the second TeacherAnnotation JSON file")
    parser.add_argument("--output", default="mergedAnnotations.csv", help="Suffix for output CSV files (default: output)")
    args = parser.parse_args()

    process_json_files(args.file1, args.file2, args.output)

    print(f"Processing complete. CSV files have been saved with suffix '{args.output}'.")

if __name__ == "__main__":
    main()
