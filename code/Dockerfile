FROM gcc

# for parallelism
ENV TBB_URL https://github.com/01org/tbb/releases/download/2017_U5/tbb2017_20170226oss_lin.tgz
RUN wget ${TBB_URL} 
RUN tar -C /opt -xf tbb2017_20170226oss_lin.tgz
RUN rm tbb2017_20170226oss_lin.tgz

# for basic python tools
RUN apt-get update
RUN apt-get -y install python-pip
RUN apt-get -y install python-dev
RUN pip install numpy
RUN pip install matplotlib

# tools for productivity
RUN apt-get -y install vim

# get starting code here 
RUN mkdir -p /scratch
RUN mkdir -p /home/CorrelationCode
RUN mkdir -p /home/CorrelationCode/testData
COPY testData/* /home/CorrelationCode/testData/
RUN mkdir -p /home/CorrelationCode/src
RUN mkdir -p /home/CorrelationCode/src/inter_results
RUN mkdir -p /home/CorrelationCode/src/params
RUN mkdir -p /home/CorrelationCode/src/params/params1
COPY src/params/params1/* /home/CorrelationCode/src/params/params1/
COPY src/* /home/CorrelationCode/src/

# so installed codes can be accessed
ENV PATH=/opt:${PATH}
