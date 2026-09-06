FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create a non-root user (uid/gid 1000 matches a typical host user, which keeps
# the bind-mounted ./data directory writable on Linux hosts as well).
RUN groupadd -g 1000 appuser && useradd -u 1000 -g 1000 -d /app -s /sbin/nologin appuser

COPY . .

# Ensure the runtime data directory is writable by the non-root user.
RUN mkdir -p /app/data && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app/ui/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
