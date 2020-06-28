#this is an attempt at creating a simple Q-Learning environment to predict stock prices of Apple
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random 


dset = pd.read_csv("aapl/dset.csv").iloc[:,1:]

closes = dset["close"].values
closes = np.append(closes, np.NaN)
closes = closes[1:]
dset["Future Closes"] = closes
dset = dset.dropna()
x = dset.iloc[:,:-1].values
y = closes[:len(closes)-1]
scX = StandardScaler()
x = scX.fit_transform(x)
scY = StandardScaler()
y = scY.fit_transform(y.reshape(-1,1))


xTrain,xTest,yTrain,yTest = train_test_split(x,y, test_size=0.2, random_state=42)



actionTable = ["UP", "DOWN"]
qTable = np.zeros([len(dset), len(actionTable)])

alpha, gamma, epsilon = 0.1, 0.6, 0.1
rewards = 0
penalties = 0

for i in range(0,len(xTrain)):
    action = random.choice(actionTable)
    result = yTrain[i]
    beginning = xTrain[i][2]
    if action == "UP" and result > beginning or action=="DOWN" and result < beginning:
        rewards += 1
    else:
        penalties+=1
print(f"Rewards: {rewards}")
print(f"Penalties: {penalties}")

