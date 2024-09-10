from flask import Flask

app = Flask(__name__, static_url_path='/static')
app.config['SECRET_KEY'] = ""
app.config['JSON_AS_ASCII'] = False
