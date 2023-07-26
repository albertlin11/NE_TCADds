import tensorflow as tf
import random
import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error as mape
from util import data_process


def mlp_model(neuron_num): #include mutation
    # mlp model configuration
    batch_size = 32
    epochs = 20000
    hidden_neurons = [neuron_num,neuron_num,neuron_num,neuron_num,neuron_num]
    layer_num = 5
    activation = "tanh"


    # data preprocess
    # train_set
    train_data = pd.read_csv("./dataset/train_set.csv")
    raw_x_train, x_train, y_train_log, y_train = data_process(train_data)

    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_scaler = scaler.fit_transform(x_train)
    scalery = MinMaxScaler(feature_range=(0, 1))
    y_train_scaler = scalery.fit_transform(y_train_log)


    # test set
    test_data = pd.read_csv("./dataset/test_set.csv")
    raw_x_test, x_test, y_test_log, y_test = data_process(test_data)

    x_test_scaler = scaler.transform(x_test)
    y_test_scaler = scalery.transform(y_test_log)



    # tensorflow setting
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=500)


    checkpoint_filepath = "./tf_model/weights"
    checkpoint_call = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    verbose=0,
    save_best_only=True,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_freq='epoch',
    )
    call_backs = [checkpoint_call, earlystopping]

    # tensorflow model
    inputs = tf.keras.Input(shape=(6, ))
    hidden = tf.keras.layers.Dense(hidden_neurons[0], activation=activation)(inputs)
    for i in range(1, layer_num):
        hidden = tf.keras.layers.Dense(hidden_neurons[i], activation=activation)(hidden)
    outdata = tf.keras.layers.Dense(1)(hidden)
    model_input = inputs
    model_output = outdata

    model = tf.keras.models.Model(inputs=model_input, outputs=model_output)
    model.compile(optimizer =opt, loss = "mse", metrics=["mape"])
    history = model.fit(x=x_train_scaler, y=y_train_scaler, validation_split=0.2, epochs= epochs,batch_size = batch_size, callbacks=call_backs, verbose = 1, shuffle=True)
    loss_info = history.history['loss']
    val_loss_info = history.history['val_loss']


    model.load_weights(checkpoint_filepath)

    model.save("./tf_model/model.h5")

    y_pred_minmax = model.predict(x_test_scaler)
    mape_minmax_scaler = mape(y_test_scaler, y_pred_minmax)

    y_test_log_inv = scalery.inverse_transform(y_test_scaler)
    y_pred_log_inv = scalery.inverse_transform(y_pred_minmax)
    mape_minmax_log_inv = mape(y_test_log_inv, y_pred_log_inv)

    y_pred = np.exp(y_pred_log_inv)
    mape_raw = mape(y_test, y_pred)

    result = np.concatenate((y_pred, y_test))
    result = result.reshape(-1, (len(y_pred_minmax)))
    result = result.transpose()
    result = pd.DataFrame(result)
    result = pd.concat([raw_x_test, result], axis=1)
    result_header = ["dopant", "Cdose1(cm^-2)","Cdose2(cm^-2)","Edose(cm^-2)", "DoseEnergy1(KeV)", "DoseEnergy2(KeV)", "prediction_IC", "IC_0.5V(Ib_2e-6) "]
    result.to_csv("process_result.csv", header=result_header)

    f = open("process_record.txt", "w")
    print(f"batch_size = {batch_size}", file=f)
    print(f"epochs = {epochs}", file=f)
    print(f"mape_minmax_scaler = {mape_minmax_scaler}", file=f)
    print(f"mape_minmax_log_inv = {mape_minmax_log_inv}", file=f)
    print(f"mape_raw = {mape_raw}", file=f)
    f.close()

    return model, loss_info, val_loss_info

def performance_mlp(num = 0):
    """It will return mape"""
    mape_raw = []
    for filename in glob.glob('*record.txt'):
        with open(os.path.join(os.getcwd(), filename), 'r') as f:
            for line in f.readlines():
                if "mape_raw " in line:
                    mape = line.replace("\n", "").replace("mape_raw = ", "")
                    mape_raw.append((filename.replace(".txt", ""),float(mape)))
            f.close()
    df_evol_results = pd.DataFrame(mape_raw)
    df_evol_results.to_csv("./evolution_log/performance/performance_mlp" + str(num) + "th_gen.csv")
    return mape_raw

# reload model
def reload_model():
    # mlp model configuration
    batch_size = 32
    epochs = 20000

    # data preprocess
    # train_set
    train_data = pd.read_csv("./dataset/train_set.csv")
    raw_x_train, x_train, y_train_log, y_train = data_process(train_data)

    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_scaler = scaler.fit_transform(x_train)
    scalery = MinMaxScaler(feature_range=(0, 1))
    y_train_scaler = scalery.fit_transform(y_train_log)


    # test set
    test_data = pd.read_csv("./dataset/test_set.csv")
    raw_x_test, x_test, y_test_log, y_test = data_process(test_data)

    x_test_scaler = scaler.transform(x_test)
    y_test_scaler = scalery.transform(y_test_log)


    # tensorflow setting
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=500)

    checkpoint_filepath = "./tf_model/weights"
    checkpoint_call = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    verbose=0,
    save_best_only=True,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_freq='epoch',
    )
    call_backs = [checkpoint_call, earlystopping]
    model = tf.keras.models.load_model("./tf_model/model.h5")
    model.fit(x=x_train_scaler, y=y_train_scaler, validation_split=0.2, epochs= epochs,batch_size = batch_size, callbacks=call_backs, verbose = 1, shuffle=True)

    model.load_weights(checkpoint_filepath)

    model.save("./tf_model/model.h5")

    y_pred_minmax = model.predict(x_test_scaler)
    mape_minmax_scaler = mape(y_test_scaler, y_pred_minmax)

    y_test_log_inv = scalery.inverse_transform(y_test_scaler)
    y_pred_log_inv = scalery.inverse_transform(y_pred_minmax)
    mape_minmax_log_inv = mape(y_test_log_inv, y_pred_log_inv)

    y_pred = np.exp(y_pred_log_inv)
    mape_raw = mape(y_test, y_pred)

    result = np.concatenate((y_pred, y_test))
    result = result.reshape(-1, (len(y_pred_minmax)))
    result = result.transpose()
    result = pd.DataFrame(result)
    result = pd.concat([raw_x_test, result], axis=1)
    result_header = ["dopant", "Cdose1(cm^-2)","Cdose2(cm^-2)","Edose(cm^-2)", "DoseEnergy1(KeV)", "DoseEnergy2(KeV)", "prediction_IC", "IC_0.5V(Ib_2e-6) "]
    result.to_csv("process_result.csv", header=result_header)

    f = open("process_record.txt", "w")
    print(f"batch_size = {batch_size}", file=f)
    print(f"epochs = {epochs}", file=f)
    print(f"mape_minmax_scaler = {mape_minmax_scaler}", file=f)
    print(f"mape_minmax_log_inv = {mape_minmax_log_inv}", file=f)
    print(f"mape_raw = {mape_raw}", file=f)
    f.close()


def run_mlp():
    for i in range(5,11):
        seed = 1
        os.environ['PYTHONHASHSEED']=str(seed)
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

        model_num = i
        model, loss_info, val_loss_info = mlp_model(model_num)
        loss_info = np.array(loss_info)
        val_loss_info = np.array(val_loss_info)
        model.save("./evolution_log/mlp_tf_model/model" + str(model_num) + ".h5")
        np.save("./evolution_log/mlp_tf_model/loss_info" + str(model_num) + ".npy", loss_info)
        np.save("./evolution_log/mlp_tf_model/val_loss_info" + str(model_num) + ".npy", val_loss_info)
        performance_mlp(model_num)

