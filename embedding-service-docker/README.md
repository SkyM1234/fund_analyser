# Embedding Service Docker

This Compose project runs the existing `milvus-docker-embedding-service:latest`
image independently from the Milvus Compose project. It does not build the
image.

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
