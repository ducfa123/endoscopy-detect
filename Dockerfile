FROM python:3.9-slim-buster
WORKDIR /app
RUN apt-get update
RUN apt-get install build-essential -y
RUN apt-get install libre2-dev git wget vim python3-enchant libenchant-dev -y
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get install libre2-5
RUN apt-get install enchant -y
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .

