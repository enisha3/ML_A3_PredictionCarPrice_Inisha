from app.app import prediction, get_X

features_val = ['1248.0', '19.391961863322244','60000.0','2015.0']

#testing if model takes the expected input
def test_get_X():
    output = get_X(*features_val)
    assert output == ('1248.0', '19.391961863322244','60000.0','2015.0'), f" Got: {output}"


#testing if the output of the model has the expected shape
def test_prediction():
    output = prediction(1248.0, 19.391961863322244, 60000.0, 2015.0)
    assert output.shape == (1,), f"Expected output shape: (1,), Got: {output.shape}"
