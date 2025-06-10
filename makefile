# Makefile for DPFL Docker workflow on Raspberry Pi 5

IMAGE_NAME = fl
DOCKERFILE = Dockerfile
MOUNT_PATH = $(shell pwd)

# Build the Docker image
build:
	docker build -f $(DOCKERFILE) -t $(IMAGE_NAME) .

# Run the container interactively with code mounted
run:
	docker run -it --rm \
		--network host \
		-v $(MOUNT_PATH):/app \
		-w /app \
		$(IMAGE_NAME) /bin/bash

# Run main.py directly (non-interactive)
start:
	docker run --rm \
		--network host \
		-v $(MOUNT_PATH):/app \
		-w /app \
		$(IMAGE_NAME) python main.py

# Rebuild from scratch (force no cache)
rebuild:
	docker build --no-cache -f $(DOCKERFILE) -t $(IMAGE_NAME) .

# Clean all containers/images (dangerous!)
clean:
	docker rmi -f $(IMAGE_NAME) || true
