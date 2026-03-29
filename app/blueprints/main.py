from flask import Blueprint, render_template


main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def home():
    return render_template("home.html")


@main_bp.route("/about")
def about():
    return render_template("about.html")


@main_bp.route("/methods")
def methods():
    return render_template("methods.html")


@main_bp.route("/documentation")
def documentation():
    return render_template("documentation.html")
