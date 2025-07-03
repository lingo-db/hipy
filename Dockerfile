FROM python:3.12-bookworm
RUN python3 --version
RUN apt-get update
RUN apt-get install -y cmake
ENV VIRTUAL_ENV=/prototyping/venv
