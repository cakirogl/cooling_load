import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, HistGradientBoostingRegressor, RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm.sklearn import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, RepeatedKFold
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from flaml import AutoML
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from time import time
plt.rcParams["text.usetex"]=True
plt.rcParams["text.latex.preamble"]=r"\usepackage{amsmath}\boldmath"
url='https://raw.githubusercontent.com/cakirogl/cooling_load/main/cooling_load_90000.csv'
df=pd.read_csv(url)
x, y = df.iloc[:, :-1], df.iloc[:, -1]
scaler=MinMaxScaler()
#x = pipeline.fit_transform(x)
#contaminations=np.arange(0.00001,0.50001,0.02)
#contaminations=[0.00001,0.02, 0.04, 0.06, 0.08,0.1, 0.18,0.30]
#contaminations=[0.00001,0.02, 0.04, 0.06, 0.08,0.1]
contaminations=[0.06]
model_selector = st.selectbox('**Cooling load prediction model**', ['XGBoost', 'LightGBM', 'CatBoost', 'Random Forest'])
input_container = st.container()
output_container = st.container()
ic1,ic2,ic3=input_container.columns(3)
with ic1:
    FA=st.number_input("**Total floor area [m^2]:**",min_value=100.0,max_value=300.0,step=10.0,value=150.0)
    AR=st.number_input("**Aspect ratio:**",min_value=0.1,max_value=1.0,step=0.05,value=0.25)
    CH=st.number_input("**Ceiling height [m]**", min_value=2.0, max_value=5.0, step=0.25, value=3.0)
    WA=st.number_input("**External wall insulation (U value) **", min_value=0.2, max_value=3.0, step=0.1, value=0.3)
with ic2:
    RO=st.number_input("**Roof insulation (U value):**", min_value=0.2, max_value=3.0, step=0.1, value=0.3)
    WI=st.number_input("**Glazing (U value):**", min_value=1.0, max_value=6.0, step=0.1, value=1.8)
    WWRN=st.number_input("**WWR north faced [%]**", min_value=10.0, max_value=90.0, step=10.0, value=10.0)
    WWRS=st.number_input("**WWR south faced [%]**", min_value=10.0, max_value=90.0, value=10.0, step=10.0)
with ic3:
    SH=st.number_input("**Horizontal shading overhang [m]:**", min_value=0.0, max_value=5.0, step=0.1, value=4.0)
    OR_=st.number_input("**Building orientation [^o]:**", min_value=0.0, max_value=360.0, step=10.0, value=90.0)

new_sample=np.array([[FA, AR, CH, WA, RO, WI, WWRN, WWRS, SH, OR_]],dtype=object)
for c in contaminations:
    model=IsolationForest(random_state=0, contamination=float(c));
    #model=IsolationForest(n_estimators=50, max_samples='auto', contamination=float(c),max_features=1.0)
    model.fit(x)
    df['scores']=model.decision_function(x)
    df['anomaly']=model.predict(x)
    anomaly_df=df.loc[df['anomaly']==-1]
    normal_df=df.loc[df['anomaly']!=-1]
    normal_data=normal_df.values
    anomaly_index=list(anomaly_df.index)
    #print(anomaly_index)
    x_normal, y_normal = normal_data[:, :-3], normal_data[:, -3]
    x_train, x_test, y_train, y_test = train_test_split(x_normal, y_normal, test_size=0.2, random_state=0)
    x_train=scaler.fit_transform(x_train)
    x_test=scaler.transform(x_test)
    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
    if model_selector=='LightGBM':
        model=LGBMRegressor(random_state=0, verbose=-1)
        model.fit(x_train,y_train)
        train_color="teal";test_color="fuchsia";eqn="-0.076+x"
    elif model_selector=='XGBoost':
        model=XGBRegressor(random_state=0)
        model.fit(x_train, y_train)
        train_color="blue";test_color="red";eqn="0.125+0.999x"
    elif model_selector=='CatBoost':
        model=CatBoostRegressor(random_state=0, logging_level="Silent")
        model.fit(x_train, y_train)
        train_color="seagreen";test_color="coral";eqn="-0.011+x"
    elif model_selector=='Random Forest':
        model=RandomForestRegressor(random_state=0, verbose=0)
        model.fit(x_train,y_train)
        train_color="royalblue";test_color="gray";eqn="-0.09+x"
    timeStart = time()
    yhat_test = model.predict(x_test)
    yhat_train = model.predict(x_train)
    automl = AutoML()
    automl_settings = {
        "estimator_list": ["catboost"],
        "metric": "r2",
        "log_training_metric": True,
        "log_type": "all",
        "model_history": True,
        "task": "regression",
        "max_iter": 50,
        "time_budget": 7000,
        "log_file_name": "logs.txt"
    }
    timeEnd=time()
    #print("Contamination = ",c);
    #print(f"Duration: {timeEnd-timeStart:.2f} seconds")
    #print("The number of anomalies:", len(anomaly_index));
    #print('MSE train= ',mean_squared_error(y_train, yhat_train))
    #print('RMSE train= ',np.sqrt(mean_squared_error(y_train, yhat_train)))
    #print('MAE train= ',mean_absolute_error(y_train, yhat_train))
    #print('R2 train:',r2_score(y_train, yhat_train))
    #print('MSE test= ',mean_squared_error(y_test, yhat_test))
    #print('RMSE test= ',np.sqrt(mean_squared_error(y_test, yhat_test)))
    #print('MAE test= ',mean_absolute_error(y_test, yhat_test))
    #print('R2 test:',r2_score(y_test, yhat_test))#original

fig, ax=plt.subplots()
with ic3:
    st.write(f"**Bond strength = **{model.predict(new_sample)[0]:.2f}** MPa**")
#ax.scatter(yhat_train, y_train, color='blue',label=r'$\mathbf{XGBoost\text{ }train}$')
#ax.scatter(yhat_test, y_test, color='red',label=r'$\mathbf{XGBoost\text{ }test}$')
#ax.scatter(yhat_train, y_train, color='seagreen',label=r'$\mathbf{CatBoost\text{ }train}$')
#ax.scatter(yhat_test, y_test, color='coral',label=r'$\mathbf{CatBoost\text{ }test}$')
#ax.scatter(yhat_train, y_train, color='teal', label=r'$\mathbf{LightGBM\text{ }train}$')
#ax.scatter(yhat_test, y_test, color='fuchsia', label=r'$\mathbf{LightGBM\text{ }test}$')
#ax.scatter(yhat_train, y_train, color='royalblue',label=r'$\mathbf{Random\text{ }Forest\text{ }train}$')
#ax.scatter(yhat_test, y_test, color='gray',label=r'$\mathbf{Random\text{ }Forest\text{ }test}$')
#ax.set_xticks([20,30,40,50,60,70])
#ax.set_xlabel(r'$\mathbf{CL_{predicted}\text{ }[kWh]}$', fontsize=14)
#ax.set_ylabel(r'$\mathbf{CL_{test}\text{ }[kWh]}$', fontsize=14)
#xmax=300;ymax=300;
#xk=[0,xmax];yk=[0,ymax];ykPlus10Perc=[0,ymax*1.1];ykMinus10Perc=[0,ymax*0.9];
#ax.tick_params(axis='x',labelsize=14)
#ax.tick_params(axis='y',labelsize=14)
#ax.plot(xk,yk, color='black')
#ax.plot(xk,ykPlus10Perc, dashes=[2,2], color='black')
#ax.plot(xk,ykMinus10Perc,dashes=[2,2], color='black')
#ax.grid(True)
#ratio=1.0
#xmin,xmax=ax.get_xlim()
#ymin,ymax=ax.get_ylim()
#ax.set_aspect(ratio*np.abs((xmax-xmin)/(ymax-ymin)));

def linearRegr(x,a0,a1):
    return a0+a1 * np.array(x)

coeffs, covmat=curve_fit(f=linearRegr, xdata=np.concatenate((yhat_train,yhat_test)).flatten(),ydata=np.concatenate((y_train,y_test)).flatten())
print(f"a0={coeffs[0]}, a1={coeffs[1]}")
#regr=linearRegr(xk,coeffs[0], coeffs[1])
#ax.plot(xk,regr, label=r"$\mathbf{y=0.125+0.999x}$")
plt.legend(loc='upper left',fontsize=12)
#plt.show()