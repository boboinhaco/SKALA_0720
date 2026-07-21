import pandas as pd

df = pd.read_csv('sales_100k')

df.shape
df.info()
df.describe()

df.isna().sum()
df.isna().sum() / len(df) * 100

df['amount'].fillna(df['amount'].median())
df['category'].fillna(df['category'].mode()[0])
df.dropna(subset=['date','amount'])

Q1 = df['amount'].quantile(0.25)
Q3 = df['amount'].quantile(0.75)
IQR = Q3 - Q1
lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
df_clean = df[df['amount'].between(lo, hi)]
print(f'이상치 {(~df['amount'].between(lo,hi)).sum()}건 체거')