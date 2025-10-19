import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train import load_data, train_model, predict, validate_data, MODEL_PATH

def test_load_and_validate_data():
    df = load_data("data/iris.csv") 
    assert validate_data(df) is True
    assert len(df) > 0

def test_model_training_and_prediction():
    df = load_data("data/iris.csv") 
    model = train_model(df)
    assert os.path.exists(MODEL_PATH)
    preds = predict(model, df)
    assert len(preds) == len(df)
    valid_species = set(df["species"])
    assert all(p in valid_species for p in preds)

if __name__ == "__main__":
    test_load_and_validate_data()
    test_model_training_and_prediction()
