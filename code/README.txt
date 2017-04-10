Step 1: create a docker image, I'll name mine test-img
docker build -t test-img .

# ideally modify step 2 to mount the test data volume. For now it just copies a folder called testData
# some modification of 
# docker run -d -v /Users/eileenmartin/Documents/SEP/campus_DAS_microseismic/auto_noise_classification/IEEEsigproc_ambientDAS/testData:/CorrelationCode/testData -p 5000:80 -it test-img

Step 2: run the docker with the data volume mounted (even if its just some test data)
docker run -i -t test-img /bin/bash

Cleanup:
docker ps (will list container ID, image name etc...)
docker stop my-container-ID-goes-here
docker rm my-container-ID-goes-here


