# PulseRank --- Adaptive YouTube Recommendation System

PulseRank is a Streamlit-based YouTube recommendation prototype that
compares a conventional popularity-ranked feed with an adaptive,
diversity-aware feed. It uses the YouTube Data API to retrieve video
metadata and demonstrates how scroll behavior can influence
recommendation ordering.

> **Project status:** Research/demo prototype. The application uses
> synthetic embeddings and heuristic ranking; it is not a trained
> personalization model or a production-scale recommendation service.

## Features

-   **YouTube search integration:** Fetches video titles, channel names,
    view counts, links, and thumbnails through the YouTube Data API.
-   **Popularity baseline:** Displays videos ranked by view count for
    comparison.
-   **Popularity categories:** Classifies retrieved videos into Head
    (popular), Mid-Tail, and Long-Tail groups using dataset-relative
    view-count percentiles.
-   **Adaptive feed prototype:** Uses scroll-speed modes to change
    category quotas and rerank the available feed.
-   **Diversity-aware reranking:** Builds a custom complex-valued
    similarity kernel and applies a greedy determinant-based selection
    strategy inspired by Determinantal Point Processes (DPPs).
-   **Live adaptation evidence:** Shows scroll telemetry and
    ranking-change information to help demonstrate when the feed
    changes.
-   **Guided demonstration:** Provides a repeatable mode-switch
    demonstration when natural scrolling does not cross a threshold.
-   **Compact Streamlit UI:** Designed as a single-viewport dashboard
    with an independently scrollable video feed.
-   **Preview fallback:** Can display clearly labelled simulated data
    when live YouTube results are unavailable, depending on the app
    configuration.

## How It Works

1.  **Retrieve videos:** The app searches YouTube for a user-provided
    query and retrieves metadata/statistics for the returned video IDs.
2.  **Categorize popularity:** View counts are compared with the 50th
    and 80th percentiles of the retrieved set.
3.  **Generate embeddings:** Each item receives a normalized
    16-dimensional complex-valued vector. The first real component is
    influenced by normalized log view count; other components are
    randomly generated. The user vector is also synthetic.
4.  **Score candidates:** A complex dot-product-derived score is
    combined with a cosine-based phase bonus.
5.  **Select and rerank:** Category quotas determine candidate pools. A
    custom complex kernel incorporates candidate quality, vector
    distance, and a phase term; a greedy determinant-based procedure
    selects the final order.
6.  **Compare feeds:** The standard feed uses popularity ordering, while
    the adaptive feed uses the custom ranking procedure.

### Important interpretation

The names "quantum," "relativistic," "UCB," and "DPP" describe
inspirations or analogies in this prototype. The implementation does
**not** simulate physical quantum processes, does not use a trained
semantic embedding model, and does not implement a conventional
uncertainty-based UCB algorithm. The Lorentz-style factor is calculated
in the original ranking design but does not directly influence the
ranking. The custom determinant procedure is DPP-inspired and should not
be treated as a formally validated DPP implementation.

## Tech Stack

-   Python
-   Streamlit
-   NumPy
-   pandas
-   python-dotenv
-   Google API Python Client (YouTube Data API)
-   HTML/CSS/JavaScript embedded in the Streamlit interface for feed
    interactions
-   Matplotlib and Seaborn are included in the dependency list but may
    not be used by every version of the app

## Project Structure

A typical project layout is:

``` text
qr-recommender-system/
├── app.py
├── requirements.txt
├── README.md
└── .env                 # Local only; do not commit secrets
```

## Setup

### 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd qr-recommender-system
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with the URL of your GitHub
repository.

### 2. Create and activate a virtual environment (recommended)

**Windows PowerShell**

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS/Linux**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

Ensure `requirements.txt` contains:

``` text
streamlit
numpy
pandas
matplotlib
seaborn
python-dotenv
google-api-python-client
```

### 4. Configure the YouTube API key

Enable **YouTube Data API v3** in your Google Cloud project and create
an API key.

For local development, create a `.env` file in the project root:

``` dotenv
YOUTUBE_API_KEY=your_actual_youtube_api_key
```

Do not commit `.env` or expose your API key in screenshots, source code,
or public logs. Add `.env` to `.gitignore`.

### 5. Run the app

``` bash
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) to
open in your browser.

## Deploying on Streamlit

1.  Push `app.py`, `requirements.txt`, and `README.md` to GitHub.
2.  Create or select the app in Streamlit Community Cloud and point it
    to the repository and entrypoint `app.py`.
3.  In the app's **Secrets** settings, add:

``` toml
YOUTUBE_API_KEY = "your_actual_youtube_api_key"
```

4.  Deploy or reboot the app after changing dependencies or secrets.

The package name is **`python-dotenv`**, while the Python import is
`from dotenv import load_dotenv`. If deployment reports
`ModuleNotFoundError: No module named 'dotenv'`, check that
`python-dotenv` is listed in `requirements.txt` at the repository root
and trigger a rebuild/redeploy.

The app may be able to run in preview mode without an API key if the
current version includes its simulated-data fallback. Preview items are
not real YouTube videos.

## Using the Feed

-   Enter a topic in the app's search control.
-   Allow the app to retrieve results or use preview data if enabled.
-   Compare the popularity-ranked feed with the adaptive feed.
-   Scroll inside the feed to generate scroll telemetry and, when
    thresholds are crossed, trigger a different ranking mode.
-   Use the guided proof/demo control to demonstrate mode changes
    consistently.
-   Review the adaptation evidence and category mix to see how the
    upcoming ranking differs.

Exact labels and controls may vary slightly by app version.

## Configuration and Data Notes

-   The YouTube API request is configured to retrieve up to 50 search
    results. YouTube may return fewer than 50 results.
-   Cached API results may not update immediately when the same query is
    repeated.
-   Popularity categories are relative to the retrieved search results,
    not all YouTube videos.
-   Scroll velocity is measured by the frontend; ranking changes occur
    when the app's configured thresholds or demonstration controls
    trigger a mode change.
-   Re-ranking and animation behavior can vary with browser size and
    device performance.

## Limitations and Future Work

-   **Synthetic personalization:** User and video embeddings are
    randomly generated, so the system does not truly understand video
    topics or individual viewing preferences.
-   **Limited catalogue:** Recommendations are based on the currently
    retrieved set (up to 50 videos), not an unlimited streaming
    catalogue.
-   **No learned feedback loop:** The prototype does not train on watch
    time, clicks, likes, or explicit user feedback.
-   **Experimental diversity method:** The custom kernel and greedy
    determinant selection need formal mathematical validation and
    quantitative evaluation before being described as a standard DPP.
-   **Evaluation needed:** Compare the methods using suitable metrics
    such as catalog coverage, intra-list diversity, NDCG,
    precision/recall where ground truth is available, and user studies.
-   **Production hardening needed:** Add robust API quota/error
    handling, tests, observability, privacy-aware event handling, and
    secure secret management before serving real users at scale.

## Troubleshooting

### `ModuleNotFoundError: No module named 'dotenv'`

Add `python-dotenv` to `requirements.txt`, commit and push the file,
then rebuild or reboot the deployment.

### YouTube API key is missing or invalid

-   Confirm the key is named `YOUTUBE_API_KEY`.
-   Check that YouTube Data API v3 is enabled in Google Cloud.
-   Verify API restrictions and quota in Google Cloud Console.
-   For deployment, set the key in Streamlit Secrets rather than relying
    on a local `.env` file.

### Fewer than 50 videos appear

The API can return fewer than the requested maximum. Try a broader
search query and check the API response and quota. Do not treat
simulated preview data as real YouTube results.

## Security

-   Never commit API keys, `.env` files, or Streamlit secrets.
-   Restrict the API key in Google Cloud where practical.
-   Do not log secrets or expose them in public error messages.

## License

Add a `LICENSE` file before publishing this repository if you want to
specify reuse terms. No license is implied by this README.
