import pickle
from pathlib import Path


def cargar_modelo_y_scaler():
    ruta_base = Path(__file__).resolve().parent.parent
    modelo_path = ruta_base / "modelo_regresion_logistica.pkl"
    scaler_path = ruta_base / "scaler.pkl"

    with modelo_path.open("rb") as modelo_file:
        modelo = pickle.load(modelo_file)

    with scaler_path.open("rb") as scaler_file:
        scaler = pickle.load(scaler_file)

    return modelo, scaler
