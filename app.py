from flask import Flask, request, render_template_string
from inferenceweb import classification_review

# ENV
LISTEN_PORT=8000
LISTEN_IP="0.0.0.0"

app = Flask(__name__)

html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Simple Bert Classification</title>
</head>
<body>
    <h2>Please enter some short text</h2>
    <form action="/" method="post">
        <textarea name="input_text" rows="4" cols="50"></textarea><br>
        <input type="submit" value="Submit">
    </form>
    <h2>Base Text:</h2>
    <p>{{ result[0] }}</p>
    <h2>Result:</h2>
    <p>{{ result[1] }}</p>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    result = ''
    if request.method == 'POST':
        input_text = request.form['input_text']
        result = classification_review(input_text)
    return render_template_string(html_template, result=result)

if __name__ == '__main__':
    app.run(host=LISTEN_IP, port=LISTEN_PORT)
