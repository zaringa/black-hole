FROM ubuntu:20.04

RUN apk update && \
    apk add python3 make

WORKDIR /app
COPY . .

RUN ln -s /app/tests /app/../tests || true 

CMD make start