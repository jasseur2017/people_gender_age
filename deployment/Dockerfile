FROM nvcr.io/nvidia/tensorrt:21.12-py3

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install -y python3-dev python3-pip ffmpeg

RUN pip3 install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN apt-get install -y sudo git vim gdb

RUN useradd -d /home/username -u 1000 -ms /bin/bash -g root -G sudo -p password username
RUN echo username:password | chpasswd

WORKDIR /home/username/people_gender_age
