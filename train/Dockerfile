FROM python:3.9

COPY trainmodel.py /app/train.py
COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

CMD ["python", "train.py"]
