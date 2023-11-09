from flask import Flask, render_template, request
import practice_beautifulsoap as data

app = Flask(__name__, template_folder='template')


@app.route('/', methods=["GET", "POST"])
def index():
    languages = data.lang()
    return render_template('index.html', languages=languages)


@app.route("/display", methods=["POST"])
def output():
    if request.method == "POST":
        entered_topic = request.form.get("topic")
        selected_language = request.form.get("language")

        soup_data = data.data(entered_topic, selected_language)
        soup_image = data.get_image_urls(entered_topic)

        return render_template('output.html', heading=entered_topic.upper(), data=soup_data,
                               url=soup_image, language=selected_language)


if __name__ == "__main__":
    app.run(debug=True)
