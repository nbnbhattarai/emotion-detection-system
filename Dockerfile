FROM python:latest
ADD . /usr/flask_app
WORKDIR /usr/flask_app/src/
EXPOSE 5000
RUN pip install --upgrade pip
RUN pip install -r api/requirements.txt
CMD systemctl start mongod

ENTRYPOINT ["python", "run_flask.py"]
