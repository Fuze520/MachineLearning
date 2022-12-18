import pandas as pd


def process_iris_data_to_csv(f_input, f_output, cols):
    with open(f_input, "r") as file:
        data = list()
        for line in file.readlines()[:-1]:
            data.append(line.strip().split(","))
        df_data = pd.DataFrame(data, columns=cols)
        df_data.to_csv(f_output, index=False)
