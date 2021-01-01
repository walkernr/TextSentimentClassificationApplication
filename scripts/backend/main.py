import uvicorn

if __name__ == "__main__":
    uvicorn.run('api_backend:app', host='localhost', port=8000, reload=True, log_level='info')