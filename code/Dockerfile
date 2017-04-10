FROM gcc

ENV TBB_URL https://github.com/01org/tbb/releases/download/2017_U5/tbb2017_20170226oss_lin.tgz
RUN wget ${TBB_URL} 
RUN tar -C /opt -xf tbb2017_20170226oss_lin.tgz
RUN rm tbb2017_20170226oss_lin.tgz

RUN mkdir -p /home/CorrelationCode
RUN mkdir -p /home/CorrelationCode/src
RUN mkdir -p /home/CorrelationCode/testData
COPY src/* /home/CorrelationCode/src
COPY testData/* /home/CorrelationCode/testData/

ENV PATH=/opt:${PATH}
