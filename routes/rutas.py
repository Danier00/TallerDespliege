from flask import Blueprint, render_template

from controllers.prediccion_controller import procesar_prediccion

rutas_bp = Blueprint("rutas", __name__)


@rutas_bp.route("/")
def formulario():
    return render_template("form.html")


@rutas_bp.route("/predecir", methods=["POST"])
def predecir():
    return procesar_prediccion()
