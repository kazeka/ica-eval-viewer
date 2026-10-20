FROM python:3.12-slim
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt
COPY --chown=user . .
EXPOSE 8501
CMD ["streamlit", "run", "viewer/app.py", "--server.port", "8501", "--server.address", "0.0.0.0", "--server.headless", "true", \
     "--server.enableCORS", "false", "--server.enableXsrfProtection", "false", "--server.fileWatcherType", "none"]
