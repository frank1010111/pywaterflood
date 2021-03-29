FROM nvidia/cuda:11.2.2-cudnn8-devel
#FROM debian:bullseye-slim

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y gcc wget procps net-tools  &&\
    apt-get autoclean -y && apt-get clean -y && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b  && \
    rm ~/miniconda.sh && \
    ~/miniconda3/bin/conda clean -tipsy && \
    echo ". /root/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc

ENV PATH /root/miniconda3/bin:$PATH
COPY ./requirements.yml ./requirements.yml

ARG JAXLIB_VERSION=0.1.64+cuda112
ARG JUPYTER
ENV PIP_FIND_LINKS https://storage.googleapis.com/jax-releases/jax_releases.html
RUN conda env update -n base -f requirements.yml
RUN /bin/bash -c 'if [[ -n "$JUPYTER" ]]; then conda install -c conda-forge jupyterlab=3 seaborn matplotlib; fi' && \
    pip install jax jaxlib==$JAXLIB_VERSION && \
    conda clean -afy

COPY ./ /pyCRM
ENV PYTHONPATH /pyCRM:$PYTHONPATH
WORKDIR /pyCRM
CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]