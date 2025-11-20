from flask import render_template, request
import numpy as np

from models.modelo import cargar_modelo_y_scaler


# Características en el orden exacto requerido por el modelo
FEATURES_ORDER = [
    "Age",
    "Sex",
    "Estado_Civil",
    "Ciudad",
    "Steroid",
    "Antivirals",
    "Fatigue",
    "Malaise",
    "Anorexia",
    "Liver_Big",
    "Liver_Firm",
    "Spleen_Palpable",
    "Spiders",
    "Ascites",
    "Varices",
    "Bilirubin",
    "Alk_Phosphate",
    "Sgot",
    "Albumin",
    "Protime",
    "Histology",
]


CHECKBOX_FIELDS = {
    "Sex",
    "Estado_Civil",
    "Ciudad",
    "Steroid",
    "Antivirals",
    "Fatigue",
    "Malaise",
    "Anorexia",
    "Liver_Big",
    "Liver_Firm",
    "Spleen_Palpable",
    "Spiders",
    "Ascites",
    "Varices",
    "Histology",
}

NUMERIC_FIELDS = {
    "Age",
    "Bilirubin",
    "Alk_Phosphate",
    "Sgot",
    "Albumin",
    "Protime",
}

model, scaler = cargar_modelo_y_scaler()


def procesar_prediccion():
    datos = {}

    for campo in FEATURES_ORDER:
        valor = request.form.get(campo)
        if campo in CHECKBOX_FIELDS:
            datos[campo] = 1 if valor is not None else 0
        elif campo in NUMERIC_FIELDS:
            try:
                datos[campo] = float(valor) if valor is not None else 0.0
            except ValueError:
                datos[campo] = 0.0
        else:
            datos[campo] = 0

    valores_ordenados = [datos[campo] for campo in FEATURES_ORDER]
    entrada = np.array(valores_ordenados).reshape(1, -1)
    entrada_escalada = scaler.transform(entrada)
    prediccion = model.predict(entrada_escalada)[0]

    mensaje = "El paciente vivirá" if prediccion == 1 else "El paciente no vivirá"
    return render_template("resultado.html", resultado=mensaje)
