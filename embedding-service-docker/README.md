# Embedding Service Docker

This Compose project runs the existing `milvus-docker-embedding-service:latest`
image independently from the Milvus Compose project. It does not build the
image.

Application source files in `../milvus-docker/` are mounted read-only. After
updating these files, run `docker compose restart embedding-service`. When
adding or changing mounts, run `docker compose up -d --force-recreate embedding-service`
once so Docker applies the new container configuration.

Start Milvus first:

```powershell
cd ..\milvus-docker
docker compose up -d
```

Then start embedding service:

```powershell
cd ..\embedding-service-docker
Copy-Item .env.example .env
docker compose up -d
```

The container waits for `http://milvus-standalone:9091/healthz` on the external
`milvus` Docker network before starting the embedding process.
