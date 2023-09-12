## base image
FROM ubuntu:22.04

## dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends --fix-missing \
    python3-pip

RUN pip3 install --upgrade pip

## set work dir
WORKDIR /app

## copy script
COPY . /app

## libraires
RUN pip3 install -r requirements.txt

## open port
EXPOSE 8501

## run app
CMD ["bash", "./start.sh"]