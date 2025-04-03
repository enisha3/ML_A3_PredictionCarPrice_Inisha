import os
import cloudpickle
import numpy as np

base_path = os.path.dirname(os.path.abspath(__file__))

model = cloudpickle.load(open(os.path.join(base_path, "./model/carprice_prediction_a3.model"), 'rb'))

def test_model_accepts_input():
    """Test if the model accepts input and does not throw an error"""
    try:
        data = np.array([[1000, 35, 5000, 2015]])
        np.exp(model.predict(data))
        passed = True
    except Exception as e:
        passed = False
    assert passed, "Model failed to accept input format"


def test_model_output_shape():
    """Test if the model output shape is (1,)"""
    data = np.array([[1000, 35, 5000, 2015]])
    prediction = np.exp(model.predict(data))
    assert prediction.shape == (1,), f"Expected shape (1,), but got {prediction.shape}"
