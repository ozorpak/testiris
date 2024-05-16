FROM python:3.9

COPY testmodel.py /app/testmodel.py
COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

CMD ["python", "testmodel.py"]
