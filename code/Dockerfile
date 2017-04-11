FROM gcc

ENV TBB_URL https://github.com/01org/tbb/releases/download/2017_U5/tbb2017_20170226oss_lin.tgz
RUN wget ${TBB_URL} 
RUN tar -C /opt -xf tbb2017_20170226oss_lin.tgz
RUN rm tbb2017_20170226oss_lin.tgz

ENV BOOST_URL https://sourceforge.net/projects/boost/files/boost/1.63.0/boost_1_63_0.tar.gz
RUN wget ${BOOST_URL}
RUN tar -C /opt -xf boost_1_63_0.tar.gz
RUN rm boost_1_63_0.tar.gz

RUN mkdir -p /home/CorrelationCode
RUN mkdir -p /home/CorrelationCode/testData
COPY testData/* /home/CorrelationCode/testData/
RUN mkdir -p /home/CorrelationCode/src
COPY src/* /home/CorrelationCode/src/


ENV PATH=/opt:${PATH}
