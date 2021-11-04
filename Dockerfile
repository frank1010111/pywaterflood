FROM debian:bullseye-slim

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y gcc wget procps net-tools git &&\
    apt-get autoclean -y && apt-get clean -y && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh -O ./miniconda.sh && \
    /bin/bash miniconda.sh -b -p/opt/conda  && \
    rm miniconda.sh && \
    /opt/conda/bin/conda clean -ay && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

ENV PATH /opt/conda/bin:$PATH
COPY ./requirements.yml ./requirements.yml
RUN conda env update -n base -f requirements.yml
RUN pip install git+https://github.com/frank1010111/pywaterflood
