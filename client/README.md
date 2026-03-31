## Fake Review Detection Frontend

This is the Next.js frontend for the Fake Review Detection Application.

It communicates with the FastAPI backend in `server/`, which loads ML models downloaded from a Google Drive folder (see the root `README.md` for full backend and model setup).

## Getting Started

Make sure you have:

- **Backend running** on `http://127.0.0.1:8000` (with models downloaded via the Drive link configured in `server/.env`).

Then install dependencies and run the development server:

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser to use the app.
