FROM python:3.11-slim AS builder

RUN apt update && apt install -y curl build-essential
RUN curl https://sh.rustup.rs -sSf | sh -s -- --default-toolchain stable -y
ENV PATH="$PATH:/root/.cargo/bin/"
RUN pip install --no-cache-dir --upgrade pip
RUN mkdir /src
COPY . /src/
RUN cd /src && \
    pip install --user --no-cache-dir .

FROM python:3.11-slim
RUN groupadd -r pywaterflood
RUN useradd -r -g pywaterflood pywaterflood
USER pywaterflood

# copy only Python packages to limit the image size
COPY --from=builder --chown=pywaterflood:pywaterflood /root/.local /home/pywaterflood/.local
ENV PATH=/home/pywaterflood/.local/bin:$PATH
