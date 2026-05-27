# Ubuntu + PPA sumo/stable (SUMO >= 1.18). Debian bookworm из python-образа даёт только SUMO 1.12.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    SUMO_HOME=/usr/share/sumo \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

ENV PATH="/opt/venv/bin:${SUMO_HOME}/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        software-properties-common \
        ca-certificates \
        curl \
    && add-apt-repository -y ppa:sumo/stable \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        sumo \
        sumo-tools \
        sumo-doc \
        python3 \
        python3-pip \
        python3-venv \
        gdal-bin \
        libgdal-dev \
        libgeos-dev \
        libproj-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && SUMO_VER="$(sumo --version | head -1 | sed -n 's/.*Version \([0-9][0-9.]*\).*/\1/p')" \
    && echo "SUMO ${SUMO_VER}" \
    && /opt/venv/bin/pip install --no-cache-dir \
        "traci==${SUMO_VER}" \
        "sumolib==${SUMO_VER}" \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt \
    && /opt/venv/bin/python -c "\
import traci; \
assert hasattr(traci.edge, 'getLastStepPersonIDs'), 'edge.getLastStepPersonIDs missing'; \
print('TraCI OK:', getattr(traci, '__version__', 'n/a'))"

COPY . .

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
