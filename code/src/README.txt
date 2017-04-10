Step 1: create a docker image, I'll name mine test-img
docker build -t test-img .

Step 2: run the docker with the data volume mounted (even if its just some test data)
docker run -it --name test-img -v /Users/eileenmartin/Documents/SEP/campus_DAS_microseismic/auto_noise_classification/IEEEsigproc_ambientDAS/testData:/CorrelationCode/testData -p 5000:80 -i /bin/bash

docker run -d -v /Users/eileenmartin/Documents/SEP/campus_DAS_microseismic/auto_noise_classification/IEEEsigproc_ambientDAS/testData:/CorrelationCode/testData -p 5000:80 -it test-img



Cleanup:
docker ps (will list container ID, image name etc...)
docker stop my-container-IDs-name-goes-here
docker rm my-container-IDs-name-goes-here


