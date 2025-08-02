FROM python:3.10-slim-bookworm

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    wget \
    python3-dev \
    autoconf \
    libtool \
    && rm -rf /var/lib/apt/lists/*

# Build and install TA-Lib C library
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

RUN ldconfig

# Copy requirements.txt first for build cache optimization
COPY requirements.txt .

# Upgrade pip and install numpy/ta-lib BEFORE other requirements for compatibility
RUN pip install --upgrade pip
RUN pip install numpy==1.25.2 ta-lib==0.6.1

# Now install the rest of your dependencies (these won't override pinned numpy)
RUN pip install -r requirements.txt

# Copy the rest of your app
COPY . .

CMD ["python", "quant_signal_bot.py"]
