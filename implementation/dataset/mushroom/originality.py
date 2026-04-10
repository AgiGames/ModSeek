import pandas as pd
pd.set_option('display.max_columns', None)

qs = pd.read_csv('questions.csv')
sols = pd.read_csv('solutions.csv')

print(f'Distinct Qs: {qs.drop_duplicates().shape[0]}')
print(f'Distinct Sols: {sols.drop_duplicates().shape[0]}')