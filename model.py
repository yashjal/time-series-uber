import pandas as pd
from fbprophet import Prophet

#read
df = pd.read_csv('final-2.csv')
df=df.drop(['Unnamed: 0'],axis=1)

#initiate
m = Prophet()

cols=list(df.columns.values)[2:]
for col in cols:
    m.add_regressor(col)

train=df.loc[0:(1322691-240522-1),:]

#valid=all 240522 entries from last day, i.e. June 2015
cols=list(df.columns.values)
cols.remove('y')
valid=df.loc[(1322691-240522):,cols]

#train/fit model
m.fit(train)
forecast = m.predict(valid)

#save predictions
forecast.to_csv('preds.csv')

m.plot(forecast)
m.plot_components(forecast)