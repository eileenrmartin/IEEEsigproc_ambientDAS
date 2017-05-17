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

6. sftp test-img.img into /data/biondo/ermartin folder on cees-mazama

7. now interact with the container through a shell (/data/biondo/DAS is a path on cees-mazama and /data is a directory that is made in the Dockerfile):
singularity shell -B /data/biondo/DAS:/data -B /scratch/ermartin:/scratch -B /scratch/fantine/das:/mnt /data/biondo/ermartin/test-img.img

8. for some reason the environment variables only get set properly to use the tbb library if you switch to bash, so just type bash after you get into the singularity image (also nice that it lets you do tab completion and up arrow for old commands) 

Type exit when you're done. Type it again to exit singularity all together. 
