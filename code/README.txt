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



============================= WITH SINGULARITY ON CEES ======================
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

Also check the size of test-img.tar and use that in step 4 (test-img.img must be a little bigger than test-img.tar).

4. create an empty singularity image
singularity create --size 2500 test-img.img
(2500 means 2500 MiB, make this bigger if you get errors in the next step saying something about not having space on device)

5. import the docker image into your empty Singularity container
singularity import test-img.img test-img.tar

(optional) If you feel like it, test out test-img.img by logging into a shell before moving it onto cees:
singularity shell test-img.img
Type exit when you're done looking around in there.

6. sftp test-img.img into your IEEEsigproc_ambientDAS/code folder on cees-mazama

7. now interact with the container through a shell (/data/biondo/DAS is a path on cees-mazama and /data is a directory that is made in the Dockerfile):
singularity shell -B /data/biondo/DAS:/data test-img.img
type exit when you're done.