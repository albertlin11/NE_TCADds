import pandas as pd
import shutil

result = pd.read_csv("result_v4.csv", index_col=False)
# result_144 = pd.read_csv("result_0501.csv", index_col=False)
seed = 1

# v4_version and v5
# delete
dopant_value = [1, 2]
cdose1_values = [1E15, 5E15]
doseenergy1_values = [40, 80, 120, 160, 200, 240, 280, 320, 360, 400]
doseenergy2_values = [30, 40, 50, 60, 70, 80, 90, 100]
dopant_filter = result["dopant"].isin(dopant_value)
cdose1_filter = result["Cdose1(cm^-2)"].isin(cdose1_values)
doseenergy1_filter = result["DoseEnergy1(KeV)"].isin(doseenergy1_values)
doseenergy2_filter2 = result["DoseEnergy2(KeV)"].isin(doseenergy2_values)
dataset = result[dopant_filter & cdose1_filter & doseenergy1_filter & doseenergy2_filter2]
dataset_drop = dataset.head(288)# 1920 2880
dataset = dataset.drop(index=dataset_drop.index)
# # register v5
# dopant_value = [1]
# cdose1_values = [1E15, 5E15]
# cdose2_values = [5E20]
# doseenergy1_values = [320, 360, 400]
# doseenergy2_values = [30,40,50,60,70, 80, 90, 100]
# dopant_filter = result["dopant"].isin(dopant_value)
# cdose1_filter = result["Cdose1(cm^-2)"].isin(cdose1_values)
# cdose2_filter = result["Edose(cm^-2)"].isin(cdose2_values)
# doseenergy1_filter = result["DoseEnergy1(KeV)"].isin(doseenergy1_values)
# doseenergy2_filter2 = result["DoseEnergy2(KeV)"].isin(doseenergy2_values)
# register = result[dopant_filter & cdose1_filter & cdose2_filter & doseenergy1_filter & doseenergy2_filter2]
# dataset = pd.concat([dataset, register, result_144])


dataset = dataset.sample(frac=1, random_state=seed)


# random
test_set = dataset.sample(frac=0.5, random_state=seed)
train_set = dataset.drop(index=test_set.index)

test_set.to_csv("test_set.csv", index=False)
train_set.to_csv("train_set.csv", index=False)
dataset = dataset.sort_values(by=['dopant','Cdose1(cm^-2)','Cdose2(cm^-2)', 'Edose(cm^-2)', 'DoseEnergy1(KeV)', 'DoseEnergy2(KeV)'])
dataset.to_csv("dataset.csv", index=False)

shutil.copy("test_set.csv", "../train_model/dataset/test_set.csv")
shutil.copy("train_set.csv", "../train_model/dataset/train_set.csv")
shutil.copy("dataset.csv", "../train_model/dataset/dataset.csv")


