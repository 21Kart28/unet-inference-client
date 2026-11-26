# Serves the fake MONAI Label endpoints so the client can be tried with no MONAI install.
# Build and run from the repository root:
#   docker compose up mock-server
FROM python:3.13-slim
WORKDIR /srv
COPY tests/mock_server.py .
EXPOSE 8000
ENTRYPOINT ["python", "mock_server.py", "--host", "0.0.0.0", "--port", "8000"]
CMD ["--cases", "5"]
