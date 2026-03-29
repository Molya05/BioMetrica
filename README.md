# Flask Biometric Recognition Project

This project already uses a Flask application factory via `create_app()` and is ready to run on Render with Gunicorn.

## Local development

Install dependencies and run the app:

```bash
pip install -r requirements.txt
python app.py
```

The local server listens on `http://127.0.0.1:8000` by default.

## Render deployment

### 1. Create the Render services

Create:

- one **Web Service** for this Flask app
- one **PostgreSQL** database for production data

You can deploy either with the included `render.yaml` or by creating the service in the Render dashboard manually.

### 2. Build and start commands

Use these commands on Render:

```text
Build Command: pip install -r requirements.txt
Start Command: gunicorn app:app
```

`app:app` works because the project root `app.py` exposes:

```python
app = create_app()
```

### 3. Required environment variables

Set these in Render:

```text
SECRET_KEY=<strong-random-secret>
DATABASE_URL=<Render Postgres Internal Database URL>
PYTHON_VERSION=3.11.11
SESSION_COOKIE_SECURE=1
REMEMBER_COOKIE_SECURE=1
PREFERRED_URL_SCHEME=https
```

Optional:

```text
DEFAULT_ADMIN_USERNAME=admin
DEFAULT_ADMIN_EMAIL=admin@example.com
DEFAULT_ADMIN_PASSWORD=<change-this>
STORAGE_ROOT=/opt/render/project/src/instance/storage
UPLOAD_FOLDER=/opt/render/project/src/instance/storage/uploads
PROCESSED_FOLDER=/opt/render/project/src/instance/storage/uploads/processed
```

Notes:

- `DATABASE_URL` is read from the environment and normalized for PostgreSQL on Render.
- Secure cookies default to enabled when the app detects Render, but explicitly setting them is still recommended.
- The app trusts proxy headers with `ProxyFix`, so HTTPS and forwarded host handling work behind Render.

### 4. Persistent uploads

This project stores captured images and processed files on disk. On Render, that means:

- if you want uploads to persist across deploys and restarts, attach a **persistent disk**
- mount it to the same path used by `STORAGE_ROOT`

Recommended mount path:

```text
/opt/render/project/src/instance/storage
```

Without a persistent disk, uploaded images and processed results will be lost on redeploy or restart.

### 5. Deploy using `render.yaml`

This repository includes a `render.yaml` file for the web service. After pushing the project to GitHub:

1. Open Render.
2. Create a new Blueprint instance from the repository.
3. Review the generated web service.
4. Add a PostgreSQL database in Render.
5. Set `DATABASE_URL` to the database internal URL if you are not wiring it automatically.
6. Set a real `SECRET_KEY`.
7. Deploy.

### 6. Verify after deployment

After the Render deploy finishes:

1. Open the Render service URL over `https://`.
2. Confirm login and registration work.
3. Confirm camera access works from a phone browser over HTTPS.
4. Confirm uploads appear under the mounted storage path.
5. Confirm the admin account uses non-default credentials.

## Production notes

- Passwords are stored as Werkzeug password hashes.
- Debug mode is disabled by default.
- Uploaded file size is capped at 8 MB.
- OpenCV uses the headless build for server deployment.
- Gunicorn is included for production serving.
- PostgreSQL is supported through `psycopg2-binary`.

## ngrok for local phone testing

If you want secure phone testing before Render deployment:

1. Start the app:

```bash
python app.py
```

2. Expose port `8000`:

```bash
ngrok http 8000
```

3. Open the generated `https://` ngrok URL on the phone.

Phone camera access requires HTTPS, so do not use plain local `http://` URLs on the phone.
