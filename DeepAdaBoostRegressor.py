import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostRegressor
import tensorflow as tf
import keras
from tensorflow.keras.optimizers import Adam
from keras import Sequential, layers
from XTilda import Memorize

class DeepAdaBoostTrader(Memorize):

  def __init__(self, path_list, n_estimators=100, epochs=20000, lr=0.05):

    super().__init__(d=0.5, level=10)

    self.half_path, self.one_path, self.two_path, self.four_path, self.eight_path = path_list
    self.n_estimators = n_estimators
    self.epochs = epochs
    self.lr = lr


  def _data_adjustment(self, df):

    df.columns = ["time", "open", "high", "low", "close", "volume"]
    df.index = pd.to_datetime(df["time"], unit="ms")
    df.drop("time", axis=1, inplace=True)

    return df

  def _extra_features(self, df):

    df["ho1"] = df["high"] / df["open"]
    df["hl1"] = df["high"] / df["low"]
    df["hc1"] = df["high"] / df["close"]
    df["lo1"] = df["low"] / df["open"]
    df["lc1"] = df["low"] / df["close"]
    df["oc1"] = df["open"] / df["close"]

    df["ho2"] = df["high"].shift(periods=1) / df["open"].shift(periods=1)
    df["hl2"] = df["high"].shift(periods=1) / df["low"].shift(periods=1)
    df["hc2"] = df["high"].shift(periods=1) / df["close"].shift(periods=1)
    df["lo2"] = df["low"].shift(periods=1) / df["open"].shift(periods=1)
    df["lc2"] = df["low"].shift(periods=1) / df["close"].shift(periods=1)
    df["oc2"] = df["open"].shift(periods=1) / df["close"].shift(periods=1)

    df["ho3"] = df["high"].shift(periods=2) / df["open"].shift(periods=2)
    df["hl3"] = df["high"].shift(periods=2) / df["low"].shift(periods=2)
    df["hc3"] = df["high"].shift(periods=2) / df["close"].shift(periods=2)
    df["lo3"] = df["low"].shift(periods=2) / df["open"].shift(periods=2)
    df["lc3"] = df["low"].shift(periods=2) / df["close"].shift(periods=2)
    df["oc3"] = df["open"].shift(periods=2) / df["close"].shift(periods=2)

    return df

  def _labeling(self, df):

    df["target"] = df["close"].shift(periods=-1)
    df.iloc[-1, -1] = df.iloc[-2, -1]

    return df

  def _tilda_version(self, df):

    df = self.x_tilda(df, "open")
    df = self.x_tilda(df, "high")
    df = self.x_tilda(df, "low")
    df = self.x_tilda(df, "close")
    df = self.x_tilda(df, "volume")

    return df

  def _preprocessing(self, df):

    adj_df = self._data_adjustment(df)
    featurerized_df = self._extra_features(adj_df)
    tilda_df = self._tilda_version(featurerized_df)

    return self._labeling(tilda_df)

  def _load_data(self):

    self.half = self._preprocessing(pd.read_json(self.half_path))
    self.one = self._preprocessing(pd.read_json(self.one_path))
    self.two = self._preprocessing(pd.read_json(self.two_path))
    self.four = self._preprocessing(pd.read_json(self.four_path))
    self.eight = self._preprocessing(pd.read_json(self.eight_path))

  def _train_test_split(self, n=0.7107):

    idx = self.eight.iloc[int(n * len(self.eight)):, :].index[0]
    start = self.eight.index[0]
    end = self.eight.index[-1]

    self.half_train = self.half.loc[start:idx, :]
    self.one_train = self.one.loc[start:idx, :]
    self.two_train = self.two.loc[start:idx, :]
    self.four_train = self.four.loc[start:idx, :]
    self.eight_train = self.eight.loc[start:idx, :]

    self.half_test = self.half.loc[idx:end, :]
    self.one_test = self.one.loc[idx:end, :]
    self.two_test = self.two.loc[idx:end, :]
    self.four_test = self.four.loc[idx:end, :]
    self.eight_test = self.eight.loc[idx:end, :]

  def _adaboost(self, df):

    X = df.iloc[:, :-1]
    Y = df.iloc[:, -1]

    adab_model = AdaBoostRegressor(n_estimators=self.n_estimators)
    adab_model.fit(X, Y)

    return (adab_model, adab_model.predict(X))

  def _apply_adaboost(self):

    self.models = []
    self.NN_inputs = pd.DataFrame(None)

    model, pre = self._adaboost(self.half_train)
    self.models.append(model)
    idx = self.half_train.index
    indexes = np.where(idx.hour + idx.minute + idx.second == 0)[0][1:]
    input_0 = pd.DataFrame(None)
    input_0["y_pre_0"] = pre[indexes]
    input_0.index = self.half_train.iloc[indexes, :].index


    model, pre = self._adaboost(self.one_train)
    self.models.append(model)
    idx = self.one_train.index
    indexes = np.where(idx.hour + idx.minute + idx.second == 0)[0][1:]
    input_1 = pd.DataFrame(None)
    input_1["y_pre_1"] = pre[indexes]
    input_1.index = self.one_train.iloc[indexes, :].index

    model, pre = self._adaboost(self.two_train)
    self.models.append(model)
    idx = self.two_train.index
    indexes = np.where(idx.hour + idx.minute + idx.second == 0)[0][1:]
    input_2 = pd.DataFrame(None)
    input_2["y_pre_2"] = pre[indexes]
    input_2.index = self.two_train.iloc[indexes, :].index

    model, pre = self._adaboost(self.four_train)
    self.models.append(model)
    idx = self.four_train.index
    indexes = np.where(idx.hour + idx.minute + idx.second == 0)[0][1:]
    input_3 = pd.DataFrame(None)
    input_3["y_pre_3"] = pre[indexes]
    input_3.index = self.four_train.iloc[indexes, :].index
    input_3.drop(["2018-02-10 00:00:00", "2018-02-11 00:00:00"], axis = 0, inplace=True)

    model, pre = self._adaboost(self.eight_train)
    self.models.append(model)
    idx = self.eight_train.index
    indexes = np.where(idx.hour + idx.minute + idx.second == 0)[0][1:]
    input_4 = pd.DataFrame(None)
    input_4["y_pre_4"] = pre[indexes]
    input_4.index = self.eight_train.iloc[indexes, :].index
    input_4.drop(["2018-02-10 00:00:00", "2018-02-11 00:00:00"], axis = 0, inplace=True)

    self.NN_inputs["in0"] = input_0["y_pre_0"]
    self.NN_inputs["in1"] = input_1["y_pre_1"]
    self.NN_inputs["in2"] = input_2["y_pre_2"]
    self.NN_inputs["in3"] = input_3["y_pre_3"]
    self.NN_inputs["in4"] = input_4["y_pre_4"]
    self.NN_inputs.index = input_4.index
    self.NN_inputs["target"] = self.eight_train.loc[self.NN_inputs.index,"target"]

  def _NN_Regressor(self, pretrained_model=False, path=None):

    x_train = tf.constant(self.NN_inputs.iloc[:, :-1])
    y_train = tf.constant(self.NN_inputs.iloc[:, -1])
    
    if pretrained_model:
    
      self.Meta_model = keras.models.load_model(path)
      self.Meta_model.fit(x=x_train, y=y_train,
                                    batch_size=64, epochs=self.epochs,
                                    shuffle=False)

    else:

      self.Meta_model = Sequential()
      self.Meta_model.add(layers.Dense(4, activation="relu"))
      self.Meta_model.add(layers.Dense(3, activation="relu"))
      self.Meta_model.add(layers.Dense(2, activation="relu"))
      self.Meta_model.add(layers.Dense(1))
    
      opt = Adam(learning_rate=self.lr)

      self.Meta_model.compile(optimizer=opt, loss="mse", metrics=["mae", "mse"])
      
      
      history = self.Meta_model.fit(x=x_train, y=y_train,
                                    batch_size=64, epochs=self.epochs,
                                    shuffle=False)
    
      plt.plot(history.history['loss'])
      plt.show()

  def fit(self, pretrained_model=False, path=None):

    self._load_data()
    self._train_test_split()
    self._apply_adaboost()
    self._NN_Regressor(pretrained_model=pretrained_model, path=path)

  def predict(self):

    self.inp_pre = pd.DataFrame(None)

    pre = self.models[0].predict(self.half_test.iloc[:, :-1])
    idx = self.half_test.index
    indexes = np.where(idx.hour + idx.minute + idx.second == 0)[0][1:]
    input_pre_0 = pd.DataFrame(None)
    input_pre_0["y_hat_pre_0"] = pre[indexes]
    input_pre_0.index = self.half_test.iloc[indexes, :].index

    pre = self.models[1].predict(self.one_test.iloc[:, :-1])
    idx = self.one_test.index
    indexes = np.where(idx.hour + idx.minute + idx.second == 0)[0][1:]
    input_pre_1 = pd.DataFrame(None)
    input_pre_1["y_hat_pre_1"] = pre[indexes]
    input_pre_1.index = self.one_test.iloc[indexes, :].index

    pre = self.models[2].predict(self.two_test.iloc[:, :-1])
    idx = self.two_test.index
    indexes = np.where(idx.hour + idx.minute + idx.second == 0)[0][1:]
    input_pre_2 = pd.DataFrame(None)
    input_pre_2["y_hat_pre_2"] = pre[indexes]
    input_pre_2.index = self.two_test.iloc[indexes, :].index


    pre = self.models[3].predict(self.four_test.iloc[:, :-1])
    idx = self.four_test.index
    indexes = np.where(idx.hour + idx.minute + idx.second == 0)[0][1:]
    input_pre_3 = pd.DataFrame(None)
    input_pre_3["y_hat_pre_3"] = pre[indexes]
    input_pre_3.index = self.four_test.iloc[indexes, :].index


    pre = self.models[4].predict(self.eight_test.iloc[:, :-1])
    idx = self.eight_test.index
    indexes = np.where(idx.hour + idx.minute + idx.second == 0)[0][1:]
    input_pre_4 = pd.DataFrame(None)
    input_pre_4["y_hat_pre_4"] = pre[indexes]
    input_pre_4.index = self.eight_test.iloc[indexes, :].index

    self.inp_pre["inp_pre_0"] = input_pre_0["y_hat_pre_0"]
    self.inp_pre["inp_pre_1"] = input_pre_1["y_hat_pre_1"]
    self.inp_pre["inp_pre_2"] = input_pre_2["y_hat_pre_2"]
    self.inp_pre["inp_pre_3"] = input_pre_3["y_hat_pre_3"]
    self.inp_pre["inp_pre_4"] = input_pre_4["y_hat_pre_4"]
    self.inp_pre.index = input_pre_4.index
    self.inp_pre["target"] = self.eight_test.loc[self.inp_pre.index,"target"]

    X_test = tf.constant(self.inp_pre.iloc[:300, :-1])
    Y_test = tf.constant(self.inp_pre.iloc[:300, -1])

    self.output = self.Meta_model.predict(X_test)

    return (self.output, Y_test)
