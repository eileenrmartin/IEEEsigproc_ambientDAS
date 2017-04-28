Step 1: create a docker image, I'll name mine test-img
docker build -t test-img .

# ideally modify step 2 to mount the test data volume. For now it just copies a folder called testData
# some modification of 
# docker run -v /Users/eileenmartin/Documents/SEP/campus_DAS_microseismic/auto_noise_classification/IEEEsigproc_ambientDAS/testData:/CorrelationCode/testData -it test-img /bin/bash

Step 2: run the docker with the data volume mounted (even if its just some test data)
docker run -i -t test-img /bin/bash

docker run -v /Users/eileenmartin/Documents/SEP/campus_DAS_microseismic/auto_noise_classification/IEEEsigproc_ambientDAS/testData:/home/CorrelationCode/testData -it test-img /bin/bash

Cleanup:
docker ps (will list container ID, image name etc...)
docker stop my-container-ID-goes-here
docker rm my-container-ID-goes-here



=============================
(can do this on oas after logging in as su)
1. on local computer create a docker image and run it as a container
docker build -t test-img .
docker run -i -t test-img /bin/bash

2. in another terminal window, look up its name, for example here it's determined_shirley:
DN0a22c039:code eileenmartin$ docker ps
CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
53a8c2c5d29a        test-img            "/bin/bash"         8 seconds ago       Up 4 seconds                            determined_shirley

3. export it as a tar with the proper name in place (may be a couple GB):
docker export determined_shirley > test-img.tar

4. sftp that into your IEEEsigproc_ambientDAS/code folder on cees-mazama

5. on cees-mazama or tool-7 or tool-8 in the IEEEsigproc_ambientDAS/code folder and create an empty singularity image
singularity create test-img.img

6. import the docker image into your empty Singularity container
singularity import test-img.img test-img.tar

7. run the Singularity container with the data volume mounted
singularity run test-img.img

8. now interact with the container through a shell
singularity shell test-img.img
type exit when you're done