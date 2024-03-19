import pandas as pd
import numpy as np
import ast
from sklearn.model_selection import train_test_split
import pickle
import warnings
import torch
warnings.filterwarnings("ignore", category=ResourceWarning)

# from unxpass.datasets import PassesDataset, CompletedPassesDataset, FailedPassesDataset
# from unxpass.components import pass_selection, pass_value, pass_success
# from unxpass.datasets import PassesDataset, CompletedPassesDataset, FailedPassesDataset
from unxpass.components import pass_success, pass_selection, pass_value
from unxpass.sampler import My_Sampler
from unxpass.datasets import PassesDataset, CompletedPassesDataset, FailedPassesDataset
field_dim = (72, 108)

def pass_surface_save(path, surface):
    with open(path, 'wb') as f:
        pickle.dump(surface, f)

dataset = pd.read_csv('./metrica-data/EPV-data/all-match.csv')
dataset = dataset[dataset['eventName'] == 'Pass']
dataset['freeze_frame'] = dataset['freeze_frame'].apply(ast.literal_eval)

train_dataset = dataset[(dataset['game_id'] != 3) | (dataset['session'] != 2)]
test_dataset = dataset[(dataset['game_id'] == 3) & (dataset['session'] == 2)]


valid_dataset, test_dataset = train_test_split(test_dataset, test_size=0.5, stratify=test_dataset['accurate'])
test_dataset.to_csv('./result/test_dataset.csv',index=False)

#pass success modeling
success_train_dataset = PassesDataset(train_dataset, transform = pass_success.ToSoccerMapTensor(dim=field_dim))
success_valid_dataset = PassesDataset(valid_dataset, transform = pass_success.ToSoccerMapTensor(dim=field_dim))
success_test_dataset = PassesDataset(test_dataset, transform = pass_success.ToSoccerMapTensor(dim=field_dim))
print("pass success modeling")
model_pass_success = pass_success.PassSuccessComponent(model=pass_success.PytorchSoccerMapModel())
model_pass_success.train(success_train_dataset, success_valid_dataset)
metric_pass_success = model_pass_success.test(success_test_dataset)
print("pass success metric: ",metric_pass_success)
surface_pass_success = model_pass_success.predict_surface(success_test_dataset)
pass_surface_save('./result/pass_success_probability_surface.pkl', surface_pass_success)


#pass selection modeling
selection_train_dataset = PassesDataset(train_dataset, transform = pass_selection.ToSoccerMapTensor(dim=field_dim))
selection_valid_dataset = PassesDataset(valid_dataset, transform = pass_selection.ToSoccerMapTensor(dim=field_dim))
selection_test_dataset = PassesDataset(test_dataset, transform = pass_selection.ToSoccerMapTensor(dim=field_dim))
print("pass selection modeling")
model_pass_selection = pass_selection.PassSelectionComponent(model=pass_selection.PytorchSoccerMapModel())
model_pass_selection.train(selection_train_dataset, selection_valid_dataset)
metric_pass_selection = model_pass_selection.test(selection_test_dataset)
print("pass selection metric: ",metric_pass_selection)
surface_pass_selection= model_pass_selection.predict_surface(selection_test_dataset)
pass_surface_save('./result/pass_selection_probability_surface.pkl', surface_pass_selection)

#pass value modeling
value_train_dataset = PassesDataset(train_dataset, transform = pass_value.ToSoccerMapTensor(dim=field_dim))
value_valid_dataset = PassesDataset(valid_dataset, transform = pass_value.ToSoccerMapTensor(dim=field_dim))
value_test_dataset = PassesDataset(test_dataset, transform = pass_value.ToSoccerMapTensor(dim=field_dim))
print("pass value modeling")
model_pass_value = pass_value.PassValueComponent(model=pass_value.PytorchSoccerMapModel())
model_pass_value.train(value_train_dataset, value_valid_dataset)
metric_pass_value = model_pass_value.test(value_test_dataset)
print("pass value metric: ",metric_pass_value)
surface_pass_value = model_pass_value.predict_surface(value_test_dataset)
pass_surface_save('./result/pass_value_probability_surface.pkl', surface_pass_value)