from contextvars import copy_context
from dash._callback_context import context_value
from dash._utils import AttributeDict
import numpy as np
from dash.testing.application_runners import import_app

from app import prediction, getDefaultValue, update_output, get_X
from LogisticRegression import LogisticRegression
from LogisticRegression import Ridge
from LogisticRegression import RidgePenalty


feature_vals = ['2015.0', '1248.0', '60000.0', '19.391961863322244']

labels = ['Cheap', 'Moderate', 'Expensive', 'Very Expensive']

possible_outputs = [label for label in labels]

# testing if model takes the expected input
def test_get_X():
    output = get_X(*feature_vals)
    assert output == ('2015.0', '1248.0', '60000.0', '19.391961863322244')

# testing if the output of the model has the expected shape
def test_prediction():
    output = prediction(2015.0, 1248.0, 60000.0, 19.391961863322244)
    assert output.shape == (1, 1)

