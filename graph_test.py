import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd


def df_equal(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    """
    Compares two DataFrames for equality regardless the rows and columns order.
    :param df1: first dataframe
    :param df2: second dataframe
    :return: comparison boolean result
    """
    # Check shapes first
    if df1.shape != df2.shape:
        print(f"\033[91mDataFrames have different shapes: {df1.shape} vs {df2.shape}\033[0m")
        return False

    df1_sorted = df1.sort_index(axis=1)
    df2_sorted = df2.sort_index(axis=1)

    df1_sorted = df1_sorted.sort_values(by=df1_sorted.columns.tolist()).values
    df2_sorted = df2_sorted.sort_values(by=df2_sorted.columns.tolist()).values

    results = np.equal(df1_sorted, df2_sorted)
    result = results.all()

    if not result:
        print(f"DataFrames are not equal in rows:")
        print(f"df1: {df1_sorted[~results]}")
        print(f"df2: {df2_sorted[~results]}")

    return result


if __name__ == '__main__':
    # Adjust to your testing needs
    inputs_path = Path("./files/input")
    outputs_path = Path("./files/output")
    spark_outputs_path = Path("./files/spark_output")
    local = True
    quick_run = False  # If True does not compute outputs, but takes what is already there.
    submit_command = "venv\\Scripts\\python.exe" if local else "spark-submit"

    passed_tests = 0
    total_tests = 0

    input_files = list(inputs_path.glob("*.csv"))
    elapsed_times = {'doubling': {}, 'linear': {}}

    for input_file in input_files:
        for algorithm in ["linear", "doubling"]:
            print(f"Testing algorithm {algorithm} and {input_file}")

            output_filename = f'{input_file.stem}_output.csv'
            spark_output_filename = f'{input_file.stem}_spark_{algorithm}_output.csv'

            if (spark_outputs_path / spark_output_filename).exists() and quick_run:
                print(f"Skipping {input_file} because output file {spark_output_filename} already exists.")
            else:
                start = time.time()
                ret_code = os.system(
                    f"{submit_command} main.py {algorithm} {str(input_file)} {str(spark_outputs_path / spark_output_filename)}")
                if ret_code != 0:
                    print(f"\033[91mError while executing main.py for {input_file}\033[0m")
                    total_tests += 1
                    continue

                stop = time.time()
                elapsed_times[algorithm][input_file.stem] = stop - start
                print(f"Elapsed time for {input_file.stem} and algorithm {algorithm}: {stop - start:.2f} seconds")

            spark_output_df = pd.read_csv(str(spark_outputs_path / spark_output_filename))
            networkx_output_df = pd.read_csv(str(outputs_path / output_filename))

            comparison_result = df_equal(spark_output_df, networkx_output_df)
            if comparison_result:
                print(f"\033[92mTest passed for {input_file} and algorithm {algorithm}\033[0m")
                passed_tests += 1
            else:
                print(f"\033[91mTest failed for {input_file} and algorithm {algorithm}\033[0m")

            total_tests += 1

    if passed_tests != total_tests:
        print(f"\033[91mSome tests failed, consult logs.\033[0m")

    print(f"\033[92mPassed {passed_tests} out of {total_tests} tests.\033[0m")

    # save output times to json
    with open('output_times.json', 'w') as f:
        json.dump(elapsed_times, f)
