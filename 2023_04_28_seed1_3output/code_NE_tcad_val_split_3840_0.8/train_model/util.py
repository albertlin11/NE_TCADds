import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error as mape


# data preprocess
def data_process(data):
    data = data.sample(frac=1, axis=0, random_state=1).reset_index(drop=True)
    raw_x = data[["dopant", "Cdose1(cm^-2)","Cdose2(cm^-2)","Edose(cm^-2)", "DoseEnergy1(KeV)", "DoseEnergy2(KeV)"]]
    x = np.array(raw_x)
    x = x.astype(float)
    raw_y = data[["IC_0.5V(Ib_2e-6) ", "IC_0.5V(Ib_3e-6) ", "IC_0.5V(Ib_4e-6) "]]
    y = np.array(raw_y)
    # Set the dose values and targets as log
    x[:, 1:4] = np.log(x[:, 1:4])
    y_log = np.log(y)
    return raw_x, x, y_log, y

def model2trend(NE_model, mlp_model): #include mutation
    # data preprocess
    # train_set
    train_data = pd.read_csv("./dataset/train_set.csv")
    raw_x_train, x_train, y_train_log, y_train = data_process(train_data)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit_transform(x_train)
    scalery = MinMaxScaler(feature_range=(0, 1))
    scalery.fit_transform(y_train_log)

    dataset = pd.read_csv("./dataset/dataset.csv")
    raw_x_all_data, x_all_data, y_all_data_log, y_all_data = data_process(dataset)
    # dataset
    x_all_data_scaler = scaler.transform(x_all_data)
    x_all_data_split = np.split(x_all_data_scaler, 6, axis = 1)


    # NE predict
    y_pred_minmax_NE = NE_model.predict(x_all_data_split)
    y_pred_log_inv_NE = scalery.inverse_transform(y_pred_minmax_NE)
    y_pred_NE = np.exp(y_pred_log_inv_NE)

    # MLP predict
    y_pred_minmax = mlp_model.predict(x_all_data_scaler)
    y_pred_log_inv = scalery.inverse_transform(y_pred_minmax)
    y_pred = np.exp(y_pred_log_inv)
    result = np.concatenate((y_pred_NE, y_pred, y_all_data), axis = 1)

    result = pd.DataFrame(result)
    result = pd.concat([raw_x_all_data, result], axis=1)
    result_header = ["dopant", "Cdose1(cm^-2)","Cdose2(cm^-2)","Edose(cm^-2)", "DoseEnergy1(KeV)", "DoseEnergy2(KeV)", "prediction_IC_0.5V(Ib_2e-6)_NE",  "prediction_IC_0.5V(Ib_3e-6)_NE", "prediction_IC_0.5V(Ib_4e-6)_NE","prediction_IC_0.5V(Ib_2e-6)_MLP",  "prediction_IC_0.5V(Ib_3e-6)_MLP", "prediction_IC_0.5V(Ib_4e-6)_MLP", "IC_0.5V(Ib_2e-6)",  "IC_0.5V(Ib_3e-6)", "IC_0.5V(Ib_4e-6)"]
    result.to_csv("process_result.csv", header=result_header, index=False)


def run_trend_gen(data_num):
    dataset = data_num
    num_line = 4
    generation = 10
    mlp = mlp_model_num(gen = generation)

    NE_model = tf.keras.models.load_model('evolution_log/tf_model/tf_' + str(generation) + 'th_gen/model0.h5')
    mlp_model = tf.keras.models.load_model('evolution_log/mlp_tf_model/model' + str(mlp) + '.h5')
    model2trend(NE_model , mlp_model)
    process_csv = pd.read_csv("process_result.csv")
    process_csv = process_csv.sort_values(by=['dopant','Cdose1(cm^-2)','Cdose2(cm^-2)', 'Edose(cm^-2)', 'DoseEnergy1(KeV)', "DoseEnergy2(KeV)"])
    # sorted by Dose Energy2
    temp = 0
    sample_rate = 8
    trend = pd.DataFrame()
    print(process_csv)
    for i in range(num_line):
        mape_max = 0
        for j in range(0, dataset, sample_rate):
            pred = process_csv.iloc[j:j+sample_rate][["prediction_IC_0.5V(Ib_2e-6)_MLP",  "prediction_IC_0.5V(Ib_3e-6)_MLP", "prediction_IC_0.5V(Ib_4e-6)_MLP"]]
            true = process_csv.iloc[j:j+sample_rate][["IC_0.5V(Ib_2e-6)",  "IC_0.5V(Ib_3e-6)", "IC_0.5V(Ib_4e-6)"]]
            mape_i = mape(pred, true)
            if mape_i > mape_max:
                mape_max = mape_i
                temp = j
        print(mape_max)
        trend_i = process_csv.iloc[temp: temp + sample_rate]
        process_csv = process_csv.drop(index=trend_i.index)
        dataset = len(process_csv)
        trend = pd.concat([trend, trend_i])
    trend.to_csv("../figure_and_table/trend_doseenergy2.csv", index=False)
    # sorted by Dose Energy1
    temp = 0
    sample_rate = 10
    trend = pd.DataFrame()
    process_csv = pd.read_csv("process_result.csv")
    process_csv = process_csv.sort_values(by=['dopant','Cdose1(cm^-2)','Cdose2(cm^-2)', 'Edose(cm^-2)', 'DoseEnergy2(KeV)', 'DoseEnergy1(KeV)'])
    for i in range(num_line):
        mape_max = 0
        for j in range(0, dataset, sample_rate):
            pred = process_csv.iloc[j:j+sample_rate][["prediction_IC_0.5V(Ib_2e-6)_MLP",  "prediction_IC_0.5V(Ib_3e-6)_MLP", "prediction_IC_0.5V(Ib_4e-6)_MLP"]]
            true = process_csv.iloc[j:j+sample_rate][["IC_0.5V(Ib_2e-6)",  "IC_0.5V(Ib_3e-6)", "IC_0.5V(Ib_4e-6)"]]
            mape_i = mape(pred, true)
            if mape_i > mape_max:
                mape_max = mape_i
                temp = j
        print(mape_max)
        trend_i = process_csv.iloc[temp: temp + sample_rate]
        process_csv = process_csv.drop(index=trend_i.index)
        dataset = len(process_csv)
        trend = pd.concat([trend, trend_i])
    trend.to_csv("../figure_and_table/trend_doseenergy1.csv", index=False)
    # sorted by Cdose2
    temp = 0
    sample_rate = 6
    trend = pd.DataFrame()
    process_csv = pd.read_csv("process_result.csv")
    process_csv = process_csv.sort_values(by=['dopant','Cdose1(cm^-2)', 'Edose(cm^-2)', 'DoseEnergy1(KeV)', 'DoseEnergy2(KeV)', 'Cdose2(cm^-2)'])
    for i in range(num_line):
        mape_max = 0
        for j in range(0, dataset, sample_rate):
            pred = process_csv.iloc[j:j+sample_rate][["prediction_IC_0.5V(Ib_2e-6)_MLP",  "prediction_IC_0.5V(Ib_3e-6)_MLP", "prediction_IC_0.5V(Ib_4e-6)_MLP"]]
            true = process_csv.iloc[j:j+sample_rate][["IC_0.5V(Ib_2e-6)",  "IC_0.5V(Ib_3e-6)", "IC_0.5V(Ib_4e-6)"]]
            mape_i = mape(pred, true)
            if mape_i > mape_max:
                mape_max = mape_i
                temp = j
        print(mape_max)
        trend_i = process_csv.iloc[temp: temp + sample_rate]
        process_csv = process_csv.drop(index=trend_i.index)
        dataset = len(process_csv)
        trend = pd.concat([trend, trend_i])
    trend.to_csv("../figure_and_table/trend_cdose2.csv", index=False)
    return process_csv

def gen_loss_data(generation):
    # select the mlp model which have simliar to the number of NE model parameters.
    gen = generation
    mlp = mlp_model_num(gen)
    # loss & val loss
    loss_gen = []
    val_loss_gen = []
    loss_header = []
    for i in range(1, gen + 1, 1):
        # NE loss and val loss
        loss = np.load("./evolution_log/records/loss_info" + str(i) + "th_gen.npy", allow_pickle=True)
        val_loss = np.load("./evolution_log/records/val_loss_info" + str(i) + "th_gen.npy", allow_pickle=True)
        loss_gen.append(loss[0])
        val_loss_gen.append(val_loss[0])
        loss_header.append("NE_gen_" + str(i))
    # mlp loss and val loss
    mlp_loss = np.load("./evolution_log/mlp_tf_model/loss_info" + str(mlp) + ".npy", allow_pickle=True)
    mlp_loss = mlp_loss.tolist()
    mlp_val_loss = np.load("./evolution_log/mlp_tf_model/val_loss_info" + str(mlp) + ".npy", allow_pickle=True)
    mlp_val_loss = mlp_val_loss.tolist()
    loss_gen.append(mlp_loss)
    val_loss_gen.append(mlp_val_loss)
    loss_header.append("MLP")
    
    loss_gen = pd.DataFrame(loss_gen).transpose()
    val_loss_gen = pd.DataFrame(val_loss_gen).transpose()
    loss_gen.to_csv("../figure_and_table/loss_gen_csv/loss_gen.csv", header=loss_header)
    val_loss_gen.to_csv("../figure_and_table/loss_gen_csv/val_loss_gen.csv", header=loss_header)
    # performance csv
    mlp_performance = []
    mlp_performance_i = pd.read_csv("./evolution_log/performance/performance_mlp" + str(mlp) + "th_gen.csv")
    mlp_performance.append(mlp_performance_i.iloc[0][2])
    NE_performance = []
    for i in range(1, gen + 1):
        NE_performance_i = []
        NE_performance_gen = pd.read_csv("./evolution_log/performance/performance" + str(i) + "th_gen.csv")
        for j in range(len(NE_performance_gen)):
            NE_performance_i.append(eval(NE_performance_gen.iloc[j][1])[1])
        NE_performance.append(NE_performance_i)


    NE_performance = np.array(NE_performance)
    NE_performance = NE_performance.transpose()

    NE_performance = pd.DataFrame(NE_performance)
    NE_performance.to_csv("../figure_and_table/NE_performance.csv", index=False)

    mlp_performance = np.array(mlp_performance)
    mlp_performance = mlp_performance.transpose()

    mlp_performance = pd.DataFrame(mlp_performance)
    mlp_performance.to_csv("../figure_and_table/mlp_performance.csv", index=False)
    
    # train_record
    loss_gen = pd.read_csv("../figure_and_table/loss_gen_csv/loss_gen.csv")
    val_loss_gen = pd.read_csv("../figure_and_table/loss_gen_csv/val_loss_gen.csv")
    # train and validation MSE
    mse_record = []
    epochs = []
    print(len(val_loss_gen["NE_gen_" + str(generation)].dropna(how='all')))
    print(len(val_loss_gen["MLP"].dropna(how='all')))
    epochs.append([len(val_loss_gen["NE_gen_" + str(generation)].dropna(how='all')), len(val_loss_gen["MLP"].dropna(how='all'))])

    NE_val_min = min(val_loss_gen["NE_gen_" + str(generation)])
    for i, value in enumerate(val_loss_gen["NE_gen_" + str(generation)]):
        if value == NE_val_min:
            break
    mse_record.append([loss_gen["NE_gen_" + str(generation)][i], val_loss_gen["NE_gen_" + str(generation)][i]])

    MLP_val_min = min(val_loss_gen["MLP"])
    for i, value in enumerate(val_loss_gen["MLP"]):
        if value == MLP_val_min:
            break
    mse_record.append([loss_gen["MLP"][i], val_loss_gen["MLP"][i]])

    NE_model = tf.keras.models.load_model('evolution_log/tf_model/tf_' + str(generation) + 'th_gen/model0.h5')
    mlp_model = tf.keras.models.load_model('evolution_log/mlp_tf_model/model' + str(mlp) + '.h5')

    # data preprocess
    # train_set
    train_data = pd.read_csv("./dataset/train_set.csv")
    raw_x_train, x_train, y_train_log, y_train = data_process(train_data)

    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_scaler = scaler.fit_transform(x_train)
    x_train_split = np.split(x_train_scaler, 6, axis = 1)
    scalery = MinMaxScaler(feature_range=(0, 1))
    y_train_scaler = scalery.fit_transform(y_train_log)
    # test set
    test_data = pd.read_csv("./dataset/test_set.csv")
    raw_x_test, x_test, y_test_log, y_test = data_process(test_data)

    x_test_scaler = scaler.transform(x_test)
    x_test_split = np.split(x_test_scaler, 6, axis = 1)
    y_test_scaler = scalery.transform(y_test_log)
    # train_num
    print(len(x_train))
    x_val_split = []
    train_num = int(len(x_train)*0.8)
    for i in range(6):
        x_val_split.append(x_train_split[i][train_num::])
        x_train_split[i] = x_train_split[i][0:train_num]
    y_val = y_train[train_num::]
    x_val_scaler = x_train_scaler[train_num::]


    y_train = y_train[0:train_num]
    x_train_scaler = x_train_scaler[0:train_num]
    y_train_scaler = y_train_scaler[0:train_num]
    print(f'{len(x_val_split)} and {x_val_split[0].shape}')
    print(f'{len(x_train_split)} and {x_train_split[0].shape}')

    # y_pred_minmax_train = NE_model.predict(x_train_scaler[train_num::])
    #val_NE_loss_mse = NE_model.evaluate(x_val_split, y_val_scaler)[0]
    #val_mlp_loss_mse = mlp_model.evaluate(x_val_scaler, y_val_scaler)[0]
    test_NE_loss_mse = NE_model.evaluate(x_test_split, y_test_scaler)[0]
    test_mlp_loss_mse = mlp_model.evaluate(x_test_scaler, y_test_scaler)[0]

    mse_record[0].append(test_NE_loss_mse)
    mse_record[1].append(test_mlp_loss_mse)

    mape_record = []
    # NE train val and test mape
    y_pred_minmax_train = NE_model.predict(x_train_split)
    y_pred_log_inv_train = scalery.inverse_transform(y_pred_minmax_train)
    y_pred_train = np.exp(y_pred_log_inv_train)
    mape_raw_train = mape(y_train, y_pred_train)

    y_pred_minmax_val = NE_model.predict(x_val_split)
    y_pred_log_inv_val = scalery.inverse_transform(y_pred_minmax_val)
    y_pred_val = np.exp(y_pred_log_inv_val)
    mape_raw_val = mape(y_val, y_pred_val)

    y_pred_minmax_test = NE_model.predict(x_test_split)
    y_pred_log_inv_test = scalery.inverse_transform(y_pred_minmax_test)
    y_pred_test = np.exp(y_pred_log_inv_test)
    mape_raw_test = mape(y_test, y_pred_test)
    mape_record.append([mape_raw_train, mape_raw_val, mape_raw_test])
    # mlp train val and test mape
    y_pred_minmax_train = mlp_model.predict(x_train_scaler)
    y_pred_log_inv_train = scalery.inverse_transform(y_pred_minmax_train)
    y_pred_train = np.exp(y_pred_log_inv_train)
    mape_raw_train = mape(y_train, y_pred_train)

    y_pred_minmax_val = mlp_model.predict(x_val_scaler)
    y_pred_log_inv_val = scalery.inverse_transform(y_pred_minmax_val)
    y_pred_val = np.exp(y_pred_log_inv_val)
    mape_raw_val = mape(y_val, y_pred_val)

    y_pred_minmax_test = mlp_model.predict(x_test_scaler)
    y_pred_log_inv_test = scalery.inverse_transform(y_pred_minmax_test)
    y_pred_test = np.exp(y_pred_log_inv_test)
    mape_raw_test = mape(y_test, y_pred_test)
    mape_record.append([mape_raw_train, mape_raw_val, mape_raw_test])

    count_NE = np.sum([np.prod(v.get_shape().as_list()) for v in NE_model.trainable_variables])
    count_mlp = np.sum([np.prod(v.get_shape().as_list()) for v in mlp_model.trainable_variables])

    print(f'best NE         loss        MSE = {mse_record[0][0]}')
    print(f'best NE         accuracy    MAPE = {mape_record[0][0]}')
    print(f'best NE     val loss        MSE = {mse_record[0][1]}')
    print(f'best NE     val accuracy    MAPE = {mape_record[0][1]}')
    print()
    print(f'best NE     test loss       MSE = {mse_record[0][2]}')
    print(f'best NE     test accuracy   MAPE = {mape_record[0][2]}')
    print('------------------------------------------------------')
    print(f'best MLP         loss       MSE = {mse_record[1][0]}')
    print(f'best MLP         accuracy   MAPE = {mape_record[1][0]}')
    print(f'best MLP     val loss       MSE = {mse_record[1][1]}')
    print(f'best MLP     val accuracy   MAPE = {mape_record[1][1]}')
    print()
    print(f'best MLP    test loss       MSE = {mse_record[1][2]}')
    print(f'best MLP    test accuracy   MAPE = {mape_record[1][2]}')
    print(f'best NE params = {count_NE}')
    print(f'best mlp params = {count_mlp}')
    record = {' ': ["NE", "MLP"],
            "epochs":[epochs[0][0], epochs[0][1]],
            "loss MSE": [mse_record[0][0], mse_record[1][0]],
            "accuracy MAPE": [mape_record[0][0], mape_record[1][0]],
            "val loss MSE": [mse_record[0][1], mse_record[1][1]],
            "val accuracy MAPE": [mape_record[0][1], mape_record[1][1]],
            "test loss MSE": [mse_record[0][2], mse_record[1][2]],
            "test accuracy MAPE": [mape_record[0][2], mape_record[1][2]],
            "params": [count_NE, count_mlp]}

    record = pd.DataFrame(record)
    record.to_csv("../figure_and_table/train_record.csv", index=False)
    return record


def mlp_model_num(gen):
    mlp_params = {}
    NE_model = tf.keras.models.load_model('evolution_log/tf_model/tf_' + str(gen) + 'th_gen/model0.h5')
    NE_params = np.sum([np.prod(v.get_shape().as_list()) for v in NE_model.trainable_variables])
    for i in [5,6,7,8,9,10]:
        model = tf.keras.models.load_model('evolution_log\mlp_tf_model\model' + str(i) + '.h5')
        mlp_params[i] = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
    for model_num, param in mlp_params.items():
        mlp_params[model_num] = abs(param - NE_params)
    return min(mlp_params, key=lambda x: mlp_params[x])

def count_params(gen=10):
    for i in range(10):
        model = tf.keras.models.load_model('evolution_log/tf_model/tf_' + str(gen) + 'th_gen/model' + str(i) + '.h5')
        count = np.sum([np.prod(v.get_shape().as_list()) for v in model.trainable_variables])
        print(count)