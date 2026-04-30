# Streamlit Community Cloud

This repository is ready for deployment on Streamlit Community Cloud.

## App entrypoint

- Main file: `app.py`

## Required files

- `requirements.txt`
- `packages.txt`

## User flow

After deployment, viewers can open the app and paste:

- OpenAI API key for generated answers
- Qdrant Cloud URL, API key, and collection for persisted retrieval

If no API key is provided, the app still works in retrieval-only mode.

## Suggested cloud setup

1. Connect the GitHub repo to Streamlit Community Cloud.
2. Set the main file path to `app.py`.
3. Add `OPENAI_API_KEY` in Streamlit secrets if you want generated answers by default.
4. Deploy.

## Public link

After deployment, paste the app URL into the `Live Demo` section of `README.md`.
