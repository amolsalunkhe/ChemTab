import pickle
import os

model_name = os.environ.setdefault('MODEL_NAME', 'PCDNNV2Model')

with open(f'models/best_models/{model_name}/experimentRecord', 'rb') as f:
	record = pickle.load(f)
print('best model R^2: ', record['model_R2'])
del record['history']
print(record)	
