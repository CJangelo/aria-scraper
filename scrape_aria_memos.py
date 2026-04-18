"""
Scrape all ARIA memo PDFs from the Sentinel Initiative website.

Strategy:
  1. Parse the main ARIA table page to collect all individual drug detail page URLs.
  2. Visit each detail page and find the link labeled "ARIA Memo".
  3. Download the PDF to a local folder.

Handles:
  - Duplicate drug names (appends _2, _3, etc.)
  - Retries with exponential backoff
  - Rate-limiting (polite delay between requests)
  - Logging to both console and a log file
"""

import re
import time
import logging
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL = "https://www.sentinelinitiative.org"
MAIN_PAGE = (
    f"{BASE_URL}/studies/drugs/"
    "assessing-arias-ability-evaluate-safety-concern"
)
OUTPUT_DIR = Path("aria_memos")

# Networking
REQUEST_TIMEOUT = 30          # seconds
DELAY_BETWEEN_PAGES = 1.5     # seconds – be polite to the server
DELAY_BETWEEN_DOWNLOADS = 1.0
MAX_RETRIES = 4
BACKOFF_FACTOR = 2            # exponential backoff multiplier

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

LOG_FILE = "aria_scraper.log"

logger = logging.getLogger("aria_scraper")
logger.setLevel(logging.DEBUG)

# Console handler – INFO and above
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
logger.addHandler(ch)

# File handler – full DEBUG detail
fh = logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(
    logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
)
logger.addHandler(fh)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def get_session() -> requests.Session:
    """Return a Session with default headers and retry adapter."""
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def fetch(session: requests.Session, url: str) -> requests.Response:
    """GET with retries and exponential backoff. Raises on final failure."""
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            wait = BACKOFF_FACTOR ** attempt
            logger.warning(
                "Attempt %d/%d failed for %s – %s. Retrying in %ds…",
                attempt, MAX_RETRIES, url, exc, wait,
            )
            time.sleep(wait)
    raise RuntimeError(
        f"All {MAX_RETRIES} attempts failed for {url}"
    ) from last_exc


# ---------------------------------------------------------------------------
# Scraping helpers
# ---------------------------------------------------------------------------

def get_drug_detail_urls(session: requests.Session) -> list[dict]:
    """
    Parse the main ARIA page table and return a list of dicts:
        {"name": "myqorzo-aficamten", "url": "https://..."}
    where url points to the Sentinel detail page for each drug.
    """
    logger.info("Fetching main ARIA table page…")
    resp = fetch(session, MAIN_PAGE)
    soup = BeautifulSoup(resp.text, "html.parser")

    # The drug links live in the first column of the table.  They point to
    # paths like /studies/drugs/assessing-arias-ability-evaluate-safety-concern/drug-slug
    entries = []
    seen_links = set()  # deduplicate if the page has exact duplicate rows

    for a_tag in soup.select("table a[href]"):
        href = a_tag["href"]
        # Only keep links that are children of the ARIA assessment path
        if "/assessing-arias-ability-evaluate-safety-concern/" not in href:
            continue
        full_url = urljoin(BASE_URL, href)
        if full_url in seen_links:
            continue
        seen_links.add(full_url)

        # Derive a slug from the URL's last path segment
        slug = href.rstrip("/").split("/")[-1]
        entries.append({"slug": slug, "url": full_url})

    logger.info("Found %d drug detail pages.", len(entries))
    return entries


def get_aria_memo_url(session: requests.Session, detail_url: str) -> str | None:
    """
    Visit a drug detail page and return the ARIA Memo PDF URL, or None
    if no memo link is found.
    """
    resp = fetch(session, detail_url)
    soup = BeautifulSoup(resp.text, "html.parser")

    # The detail page has links labeled "ARIA Memo" pointing to
    # accessdata.fda.gov …OtherR.pdf
    for a_tag in soup.find_all("a", href=True):
        link_text = a_tag.get_text(strip=True).lower()
        if "aria memo" in link_text:
            return a_tag["href"]

    # Fallback: look for any accessdata link ending in OtherR.pdf
    for a_tag in soup.find_all("a", href=True):
        href = a_tag["href"]
        if "accessdata.fda.gov" in href and href.lower().endswith("otherr.pdf"):
            return href

    return None


def sanitize_filename(slug: str) -> str:
    """Turn a URL slug into a clean filename stem (no extension)."""
    # Replace non-alphanumeric chars (except hyphens) with hyphens
    name = re.sub(r"[^a-zA-Z0-9\-]", "-", slug)
    # Collapse multiple hyphens
    name = re.sub(r"-{2,}", "-", name).strip("-")
    return name


def resolve_duplicate(directory: Path, stem: str, suffix: str = ".pdf") -> Path:
    """
    Return a unique file path.  If 'stem.pdf' already exists, try
    'stem_2.pdf', 'stem_3.pdf', etc.
    """
    candidate = directory / f"{stem}{suffix}"
    if not candidate.exists():
        return candidate
    counter = 2
    while True:
        candidate = directory / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    session = get_session()

    # Step 1 – collect all drug detail page URLs from the main table
    drugs = get_drug_detail_urls(session)
    if not drugs:
        logger.error("No drug entries found – page structure may have changed.")
        return

    # Step 2 & 3 – visit each detail page, find the memo link, download
    downloaded = 0
    skipped = 0
    failed = 0

    for i, drug in enumerate(drugs, start=1):
        slug = drug["slug"]
        detail_url = drug["url"]
        logger.info("[%d/%d] Processing %s", i, len(drugs), slug)

        # --- find the ARIA memo PDF URL ---
        try:
            time.sleep(DELAY_BETWEEN_PAGES)
            memo_url = get_aria_memo_url(session, detail_url)
        except Exception:
            logger.exception("  Failed to fetch detail page: %s", detail_url)
            failed += 1
            continue

        if memo_url is None:
            logger.warning("  No ARIA memo link found – skipping.")
            skipped += 1
            continue

        logger.debug("  Memo URL: %s", memo_url)

        # --- download the PDF ---
        filename_stem = sanitize_filename(slug)
        dest = resolve_duplicate(OUTPUT_DIR, filename_stem)

        try:
            time.sleep(DELAY_BETWEEN_DOWNLOADS)
            pdf_resp = fetch(session, memo_url)
        except Exception:
            logger.exception("  Failed to download PDF: %s", memo_url)
            failed += 1
            continue

        dest.write_bytes(pdf_resp.content)
        size_kb = len(pdf_resp.content) / 1024
        logger.info("  Saved %s (%.1f KB)", dest.name, size_kb)
        downloaded += 1

    # --- summary ---
    logger.info("=" * 60)
    logger.info("Done.  Downloaded: %d  |  Skipped: %d  |  Failed: %d",
                downloaded, skipped, failed)
    logger.info("PDFs saved to: %s", OUTPUT_DIR.resolve())
    logger.info("Full log: %s", Path(LOG_FILE).resolve())


if __name__ == "__main__":
    main()