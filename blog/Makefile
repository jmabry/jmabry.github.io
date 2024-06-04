DOCKER=docker

TAG=gh-pages

SITE=site

# Build the docker image
image:
	${DOCKER} build -t ${TAG} .


server:
	${DOCKER} run --rm -it \
		-p 4000:4000 \
		-u `id -u`:`id -g` \
		-v `realpath ${SITE}`:/src/site \
		-w /src/site \
		${TAG}

# terminal:
#     ${DOCKER} run --rm -it \
#         -p 4000:4000 \
#         -u `id -u`:`id -g` \
#         -v `realpath ${SITE}`:/src/site \
#         -w /src/site \
#         ${TAG} /bin/bash

.PHONY:
	image server terminal