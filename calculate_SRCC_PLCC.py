import pandas as pd
from scipy.stats import spearmanr, pearsonr




def Calculate_SRCC_PLCC(file_path):

    df = pd.read_csv(file_path)


    df['ratings'] = df['ratings'].str.strip('[]')
    df['predictions'] = df['predictions'].str.strip('[]')

    df['ratings'] = pd.to_numeric(df['ratings'], errors='coerce')
    df['predictions'] = pd.to_numeric(df['predictions'], errors='coerce')

    missing_ratings = df['ratings'].isna().sum()
    missing_predictions = df['predictions'].isna().sum()

    if missing_ratings > 0 or missing_predictions > 0:
        print(f"WARNING: There is NAN in the data - ratings: {missing_ratings}, predictions: {missing_predictions}")

        # removing NAN
        original_count = len(df)
        df = df.dropna(subset=['ratings', 'predictions'])
        removed_count = original_count - len(df)

        if removed_count > 0:
            print(f"Removed {removed_count} 行数据")

    # Check if there are any samples left in the processed data
    if len(df) == 0:
        raise ValueError("The processed data does not contain any valid samples, please check the CSV file format and contents")


    srcc, srcc_p_value = spearmanr(df['ratings'], df['predictions'])

    plcc, plcc_p_value = pearsonr(df['ratings'], df['predictions'])

    srcc_rounded = round(srcc, 2)
    srcc_p_value_rounded = f'{srcc_p_value:.2e}'
    plcc_rounded = round(plcc, 2)
    plcc_p_value_rounded = f'{plcc_p_value:.2e}'

    result = {
        'SRCC': srcc_rounded,
        'p-value of SRCC': srcc_p_value_rounded,
        'PLCC': plcc_rounded,
        'p-value of PLCC': plcc_p_value_rounded
    }
    return result


"""
Change the following file path to the file output path you specified when training the model.
"""


PRI_test_output =  "/home/cbl/IQA/zihao/test_output_dataframe_clip_5.csv"
FG_test_output  =  "/home/cbl/IQA/zihao/fg_output_dataframe_clip_5.csv"
RG1_test_output =  "/home/cbl/IQA/zihao/rg1_output_dataframe_clip_5.csv"
RG2_test_output =  "/home/cbl/IQA/zihao/rg2_output_dataframe_clip_5.csv"



test_files = [PRI_test_output, FG_test_output, RG1_test_output, RG2_test_output]





for test_file in test_files:
    print(test_file)
    correlation_results = Calculate_SRCC_PLCC(test_file)
    print(correlation_results)
    print("###################################################")