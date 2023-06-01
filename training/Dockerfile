FROM nvcr.io/nvidia/pytorch:22.07-py3

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN apt-get install -y sudo vim git wget unzip ffmpeg
# jupyter

RUN useradd -d /home/jasseur -u 1000 -ms /bin/bash -g root -G sudo -p password jasseur
RUN echo jasseur:password | chpasswd

WORKDIR /home/jasseur/people_gender_age/
