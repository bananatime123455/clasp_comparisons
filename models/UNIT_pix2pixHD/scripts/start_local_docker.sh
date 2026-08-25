docker run \
    --gpus all \
    --shm-size 32g \
    --ipc=host \
    -it \
    -v /mnt:/mnt \
    -v ~/:/home \
    docker.io/tdougherty84/imaginaire:latest \
    /bin/bash

