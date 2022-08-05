FROM python:3.9-slim

RUN pip install pipenv

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["predict.py", "model_C=10.bin", "./"]

EXPOSE 8000

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:8000", "predict:app"]