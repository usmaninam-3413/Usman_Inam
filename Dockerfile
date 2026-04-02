FROM python:3.14

WORKDIR /script

COPY script.py .

RUN pip install numpy

CMD ["python", "script.py"]