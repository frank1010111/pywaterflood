FROM python:3.9-slim AS builder

RUN mkdir /src
COPY . /src/
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    cd /src && \
    pip install --user --no-cache-dir .

FROM python:3.9-slim

# copy only Python packages to limit the image size
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
