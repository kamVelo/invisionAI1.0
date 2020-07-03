# this is an attempt at creating a simple Q-Learning environment to predict stock prices of Apple
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense

dset = pd.read_csv("aapl/dset.csv").iloc[:, 1:]
rsi = pd.read_csv("aapl/rsi.csv").iloc[:, 1:]
lengths = [len(dset), len(rsi)]
idx = min(lengths)
dset, rsi = dset[:idx], rsi[:idx]
rsiVals = []
for index, row in rsi.iterrows():
    RSI = row["RSI"]
    if RSI < 30:
        data = [1, 0, 0]
    elif RSI > 70:
        data = [0, 0, 1]
    else:
        data = [0, 0, 0]
    rsiVals.append(data)

rsiVals = pd.DataFrame(rsiVals, columns={"LOW", "MID", "HIGH"})
dset = pd.concat([dset, rsiVals], axis=1)
closes = np.append(dset["close"].values, np.NaN)
closes = closes[1:-1]

opens = np.append(dset["open"].values, np.NaN)
opens = opens[1:-1]
dset = dset[:-1]
x = dset[:-1]
polyReg = PolynomialFeatures(degree=4)
x = polyReg.fit_transform(x)
y = []
for i in range(0, len(closes) - 1):
    if closes[i] > opens[i]:
        y.append("UP")
    elif closes[i] < opens[i] or closes[i] == opens[i]:
        y.append("DOWN")

encoder = LabelEncoder()
y = encoder.fit_transform(y)
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.2, random_state=42)

scX = StandardScaler()
xTrain = scX.fit_transform(xTrain)
xTest = scX.fit_transform(xTest)

model = tf.keras.Sequential()
model = keras.Sequential([Dense(16, activation="tanh"),
                          Dense(16, activation="tanh"),
                          Dense(1, activation="tanh")])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(xTrain, yTrain, batch_size=32, epochs=1000)
pred = model.predict(xTest)
roundPred = []

for i in pred:
    avg = sum(i) / len(i)

    if avg >= 0.5:
        val = 1
    elif avg < 0.5:
        val = 0
    roundPred.append(val)
correct = 0
for i in range(0, len(roundPred)):
    if roundPred[i] == yTest[i]:
        correct += 1

print(f"Percentage Correct: {(correct / len(roundPred) * 100)}")
