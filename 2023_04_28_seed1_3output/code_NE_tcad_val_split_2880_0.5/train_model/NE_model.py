import tensorflow as tf
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error as mape
from util import data_process



#define how many layers and blocks. also give them ids
def create_blocks(layer_max, block_max):
    tot_l = random.randint(2, layer_max)
    layers = []
    for i in range(tot_l):
        blocks = random.randint(1, block_max)
        layers.append(blocks)
    k = 1
    blocks_id = []
    for i in layers:
        ids = []
        for n in range(i):
            ids.append((k, n+1))
        k+=1
        blocks_id.append(ids)
    return blocks_id


#define the connection(blocks in i layer are only allowed to connect to blocks in i-1 layer),and
#randomly assign number of neuron and activation function of each block
def create_conn(n_max,blocks_id):
    Hidden = {}
    Output = {}
    input_key = [(-1, 1),(-1, 2),(-1, 3),(-1, 4),(-1, 5),(-1, 6)]                       # modify input size
    activations = ['tanh', 'sigmoid']
    for key, value in enumerate(blocks_id):
        for i in value:
            if key == 0:
                r = random.randint(3, 6)
                conn_1 = random.sample(input_key, r)
                conn_1.sort()
                act = random.choice(activations)
                neurons = random.randint(1, n_max)
                Hidden[i] = [neurons, act,conn_1]
            else:
                l = len(blocks_id[key-1])
                r2 = random.randint(1, l)
                conn = random.sample(blocks_id[key-1], r2)
                conn.sort()
                act = random.choice(activations)
                neurons = random.randint(1, n_max)
                Hidden[i] = [neurons, act, conn]
    Output = {}
    out_blocks_l = len(blocks_id[-1])
    r_out = random.randint(out_blocks_l//2+1, out_blocks_l)
    out_conn = random.sample(blocks_id[-1], r_out)
    out_conn.sort()
    Output[(0,0)] = out_conn
    return Hidden, Output

# construct the neural-evlution architecture
def create_architecture(l_max, b_max, n_max):
    ids = create_blocks(l_max, b_max)
    hidden, output =  create_conn(n_max, ids)
    return [hidden, output]


# Build the model using Keras functional api. If the input genome has missing connection ids or
# reapeted block ids, create_model function will repair the connection and return a keras model.
def NE_model(genome): #include mutation


    # NE model configuration
    architecture = genome['process_architecture']
    batch_size = 32
    epochs = 20000

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
    Value = {}
    Value[(-1,1)] = tf.keras.Input(shape=(1,),name = 'dopant')
    Value[(-1,2)] = tf.keras.Input(shape=(1,),name = 'Cdose1(cm^-2)')
    Value[(-1,3)] = tf.keras.Input(shape=(1,),name = 'Cdose2(cm^-2)')
    Value[(-1,4)] = tf.keras.Input(shape=(1,),name = 'Edose(cm^-2)')
    Value[(-1,5)] = tf.keras.Input(shape=(1,),name = 'DoseEnergy1(KeV)')
    Value[(-1,6)] = tf.keras.Input(shape=(1,),name = 'DoseEnergy2(KeV)')
    hidden = architecture[0]
    output = architecture[1]
    layers = []
    keys = list(hidden.keys())
    keys.sort()
    for key in keys:
        layers.append(key[0])
    layers=set(layers)
    layer_ids = []
    for i in layers:
        L = []
        for k in keys:
            if k[0] == i:
                L.append(k)
        layer_ids.append(L)

    for layer, ids in enumerate(layer_ids):
        if layer == 0:
            for i, id in enumerate(ids):
                input_ids_value = []
                neuron = hidden[id][0]
                activation = hidden[id][1]
                initializer = tf.keras.initializers.GlorotUniform()
                for conn in hidden[id][2]:
                    input_ids_value.append(Value[conn])
                if len(input_ids_value) == 1:
                    inputdata = input_ids_value[0]
                else:
                    inputdata = tf.keras.layers.Concatenate()(input_ids_value)
                l = tf.keras.layers.Dense(neuron, activation, kernel_initializer=initializer)(inputdata)
                Value[id] = l

        else:
            for i in ids:
                input_ids_value = []
                neuron = hidden[i][0]
                activation = hidden[i][1]
                initializer = tf.keras.initializers.GlorotUniform()
                conns = hidden[i][2]
                for key, conn in enumerate(conns):
                    if conn not in layer_ids[layer-1]:
                        new_conn = random.choice(layer_ids[layer-1])
                        conns[key] = new_conn
                conns = list(set(conns))
                conns.sort()
                hidden[i][2] = conns
                for conn in hidden[i][2]:
                    input_ids_value.append(Value[conn])
                if len(input_ids_value) == 1:
                    l = tf.keras.layers.Dense(neuron, activation, kernel_initializer=initializer)(input_ids_value[0])
                else:
                    inputdata = tf.keras.layers.Concatenate()(input_ids_value)
                    l = tf.keras.layers.Dense(neuron, activation, kernel_initializer=initializer)(inputdata)
                Value[i]=l

    output_conn = output[(0,0)]
    for key, conn in enumerate(output_conn):
        if conn not in layer_ids[-1]:
            new_conn = random.choice(layer_ids[-1])
            output_conn[key] = new_conn
    output_conn = list(set(output_conn))
    output_conn.sort()
    output[(0,0)] = output_conn

    output_ids_value = []
    for ids in output[(0,0)]:
        output_ids_value.append(Value[ids])
    if len(output_ids_value) == 1:
        outdata = output_ids_value[0]
    else:
        outdata = tf.keras.layers.Concatenate()(output_ids_value)

    outdata = tf.keras.layers.Dense(3)(outdata)
    model_input = [Value[(-1,1)], Value[(-1,2)], Value[(-1,3)], Value[(-1,4)], Value[(-1,5)], Value[(-1,6)]]
    model_output = outdata
    model = tf.keras.models.Model(inputs=model_input, outputs=model_output)

    model.compile(optimizer =opt, loss = "mse", metrics=["mape"])
    history = model.fit(x=x_train_split, y=y_train_scaler, validation_split=0.2, epochs= epochs,batch_size = batch_size, callbacks=call_backs, verbose = 1, shuffle=True)

    loss_info = history.history['loss']
    val_loss_info = history.history['val_loss']

    model.load_weights(checkpoint_filepath)

    model.save("./tf_model/model.h5")
    trainable_params = np.sum([np.prod(weights.get_shape()) for weights in model.trainable_weights])

    val_accuracy = min(history.history['val_loss'])
    # test_mape
    y_pred_minmax_test = model.predict(x_test_split)

    y_pred_log_inv_test = scalery.inverse_transform(y_pred_minmax_test)

    y_pred_test = np.exp(y_pred_log_inv_test)
    mape_raw_test = mape(y_test, y_pred_test)

    result = np.concatenate((y_pred_test, y_test))
    result = result.reshape(-1, (len(y_pred_minmax_test)))
    result = result.transpose()
    result = pd.DataFrame(result)
    result = pd.concat([raw_x_test, result], axis=1)
    result_header = ["dopant", "Cdose1(cm^-2)","Cdose2(cm^-2)","Edose(cm^-2)", "DoseEnergy1(KeV)", "DoseEnergy2(KeV)", "prediction_IC_0.5V(Ib_2e-6) ",  "prediction_IC_0.5V(Ib_3e-6) ", "prediction_IC_0.5V(Ib_4e-6)", "IC_0.5V(Ib_2e-6) ",  "IC_0.5V(Ib_3e-6) ", "IC_0.5V(Ib_4e-6) "]
    result.to_csv("process_result.csv", header=result_header)

    f = open("process_record.txt", "w")
    print(f"batch_size = {batch_size}", file=f)
    print(f"epochs = {epochs}", file=f)
    print(f"mape_raw = {val_accuracy}", file=f)
    print(f"trainable_params = {trainable_params}", file=f)
    print(f"test_mape_performance = {mape_raw_test}", file=f)
    f.close()

    return model, loss_info, val_loss_info

# Build the model using Keras functional api. If the input genome has missing connection ids or
# reapeted block ids, create_model function will repair the connection and return a keras model.
def reload_model(model): #include mutation

    # NE model configuration
    batch_size = 32
    epochs = 20000#int(genome['process_epoch'])

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
    model.compile(optimizer =opt, loss = "mse", metrics=["mape"])
    history = model.fit(x=x_train_split, y=y_train_scaler, validation_split=0.2, epochs= epochs,batch_size = batch_size, callbacks=call_backs, verbose = 1, shuffle=True)
    loss_info = history.history['loss']
    val_loss_info = history.history['val_loss']

    model.load_weights(checkpoint_filepath)

    model.save("./tf_model/model.h5")
    trainable_params = np.sum([np.prod(weights.get_shape()) for weights in model.trainable_weights])

    val_accuracy = min(history.history['val_loss'])
    # test_mape
    y_pred_minmax_test = model.predict(x_test_split)

    y_pred_log_inv_test = scalery.inverse_transform(y_pred_minmax_test)

    y_pred_test = np.exp(y_pred_log_inv_test)
    mape_raw_test = mape(y_test, y_pred_test)

    result = np.concatenate((y_pred_test, y_test))
    result = result.reshape(-1, (len(y_pred_minmax_test)))
    result = result.transpose()
    result = pd.DataFrame(result)
    result = pd.concat([raw_x_test, result], axis=1)
    result_header = ["dopant", "Cdose1(cm^-2)","Cdose2(cm^-2)","Edose(cm^-2)", "DoseEnergy1(KeV)", "DoseEnergy2(KeV)", "prediction_IC_0.5V(Ib_2e-6) ",  "prediction_IC_0.5V(Ib_3e-6) ", "prediction_IC_0.5V(Ib_4e-6)", "IC_0.5V(Ib_2e-6) ",  "IC_0.5V(Ib_3e-6) ", "IC_0.5V(Ib_4e-6) "]
    result.to_csv("process_result.csv", header=result_header)

    f = open("process_record.txt", "w")
    print(f"batch_size = {batch_size}", file=f)
    print(f"epochs = {epochs}", file=f)
    print(f"mape_raw = {val_accuracy}", file=f)
    print(f"trainable_params = {trainable_params}", file=f)
    print(f"test_mape_performance = {mape_raw_test}", file=f)
    f.close()

    return model, loss_info, val_loss_info
