import random

import numpy as np
import pandas as pd
import datetime

df_length = 600

dates = [
    datetime.datetime.now() - datetime.timedelta(minutes=i)
    for i in range(df_length, 0, -1)
]

prices = [round(random.uniform(1, 1000), 2) for i in range(df_length)]

df = pd.DataFrame({"date": dates, "price": prices})
# print(df)

# df['minute'] = df['date'].apply(lambda x: x.minute)
df["date"] = pd.to_datetime(df["date"], unit="minute")
print(df)

df = df.set_index("date")
df = df.resample("H", axis=0).ohlc(_method="ohlc")

print(df)
