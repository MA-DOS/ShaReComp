FROM ubuntu:24.04

RUN apt-get update && \
    apt-get install -y git build-essential automake libtool pkg-config \
                       libaio-dev libmysqlclient-dev && \
    git clone --depth 1 https://github.com/akopytov/sysbench && \
    cd sysbench && \
    ./autogen.sh && ./configure && make -j && make install && \
    cd / && rm -rf /sysbench && \
    apt-get remove -y git build-essential automake libtool pkg-config \
                       libaio-dev libmysqlclient-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*