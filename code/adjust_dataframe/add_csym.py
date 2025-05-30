import pandas as pd


output_pd_path = './EMDB_validation.csv'
df = pd.read_csv(output_pd_path)

df['csym_validated'] = df['csym_validated'].apply(lambda x: f'C{x}' if not str(x).startswith('C') else x)

df.to_csv(output_pd_path, index=False)

print(df['csym_validated'])