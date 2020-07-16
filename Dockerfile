FROM tensorflow/tensorflow:1.12.0-devel-gpu-py3

RUN groupadd -g 1012 iafurger
RUN useradd -s /bin/bash -u 1012 -g 1012 -m iafurger
RUN echo 'cd $APP_DIR' >> $(su -c 'echo $HOME' iafurger)/.bashrc

RUN add-apt-repository 'deb http://ch.archive.ubuntu.com/ubuntu/ xenial main universe'
RUN apt-get update
RUN apt-get install -y less vim

RUN pip3 install --upgrade pip
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN mkdir -p venv/bin && ln -s $(which python) venv/bin/python

ENTRYPOINT /bin/bash
