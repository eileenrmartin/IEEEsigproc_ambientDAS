FROM gcc

# for parallelism
ENV TBB_URL https://github.com/01org/tbb/releases/download/2017_U5/tbb2017_20170226oss_lin.tgz
RUN wget ${TBB_URL} && tar -C /usr/local -xf tbb2017_20170226oss_lin.tgz && rm tbb2017_20170226oss_lin.tgz && mv /usr/local/tbb2017_20170226oss /usr/local/tbb

# for basic python tools
RUN apt-get update
RUN apt-get -y install python-pip
RUN apt-get -y install python-dev
RUN apt-get -y install libopenblas-dev
RUN pip install numpy scipy matplotlib
RUN pip install obspy
RUN pip install scikit-learn
RUN apt-get -y install swig


# tools for productivity
RUN apt-get -y install vim

# get starting code here 
RUN mkdir -p /scratch
RUN mkdir -p /data
RUN mkdir -p /CorrelationCode
RUN mkdir -p /CorrelationCode/testData
COPY testData/* /CorrelationCode/testData/
RUN mkdir -p /CorrelationCode/src
RUN mkdir -p /CorrelationCode/src/inter_results
COPY src/inter_results/* /CorrelationCode/src/inter_results/
RUN mkdir -p /CorrelationCode/src/params
RUN mkdir -p /CorrelationCode/src/params/params1
#COPY src/params/params1/* /CorrelationCode/src/params/params1/
COPY src/* /CorrelationCode/src/

# so installed codes can be accessed
RUN echo "source /usr/local/tbb/bin/tbbvars.sh intel64" >> /etc/bash.bashrc
ENV LD_LIBRARY_PATH=/usr/local/tbb/lib/intel64/gcc4.7:$LD_LIBRARY_PATH
ENV LIBRARY_PATH=/usr/local/tbb/lib/intel64/gcc4.7:$LIBRARY_PATH
ENV PATH=/usr/local/tbb:${PATH}
