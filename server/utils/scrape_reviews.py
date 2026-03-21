# utils/amazon_scraper.py

import time
import csv
from pathlib import Path
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from dataclasses import dataclass


# ── Config ────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "reviews"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Review:
    content: str
    rating: str


# ── Driver setup ──────────────────────────────────────────────────────────────

def build_driver() -> webdriver.Chrome:
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )
    driver.execute_cdp_cmd(
        "Page.addScriptToEvaluateOnNewDocument",
        {"source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"},
    )
    return driver


# ── Login wait ────────────────────────────────────────────────────────────────

def wait_for_login(driver: webdriver.Chrome, asin: str, timeout: int = 100) -> None:
    first_page = f"https://www.amazon.in/product-reviews/{asin}/?pageNumber=1"
    driver.get(first_page)

    wait = WebDriverWait(driver, timeout)
    try:
        wait.until(EC.text_to_be_present_in_element(
            (By.ID, "nav-link-accountList-nav-line-1"), "Hello,"
        ))
    except Exception:
        raise TimeoutError("Login not completed within the time limit.")


# ── Parsing ───────────────────────────────────────────────────────────────────

def _safe_text(element, selector: str) -> str:
    node = element.select_one(selector)
    return node.get_text(strip=True) if node else ""


def _parse_rating(rating: str) -> float:
    try:
        return float(rating.split(" ")[0])
    except:
        return 0.0


def parse_reviews(soup: BeautifulSoup) -> list[Review]:
    return [
        Review(
            content = _safe_text(r, "[data-hook='review-body']"),
            rating  = _safe_text(r, "[data-hook='review-star-rating']"),
        )
        for r in soup.select("li[data-hook='review']")
    ]


# ── Scrape one filter ─────────────────────────────────────────────────────────

def scrape_filter(
    driver: webdriver.Chrome,
    asin: str,
    params: str,
    pages: int,
    delay: float,
    seen: set,
) -> list[Review]:

    wait = WebDriverWait(driver, 10)
    new_reviews = []

    for page in range(1, pages + 1):
        url = f"https://www.amazon.in/product-reviews/{asin}/?{params}&pageNumber={page}"
        driver.get(url)

        try:
            wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, "li[data-hook='review']")
            ))
        except Exception:
            break

        soup = BeautifulSoup(driver.page_source, "html.parser")

        added = 0
        for r in parse_reviews(soup):
            key = (r.content, r.rating)
            if key not in seen:
                seen.add(key)
                new_reviews.append(r)
                added += 1

        if added == 0:
            break

        time.sleep(delay)

    return new_reviews


# ── Main utility function ─────────────────────────────────────────────────────

def scrape_and_save(
    asin: str,
    pages: int = 5,
    delay: float = 3.0,
    login_timeout: int = 100,
) -> list[list]:
    """
    Scrapes Amazon reviews for the given ASIN across 5 filters.
    Returns a list of [content, rating] pairs where rating is a float.
    e.g. [["Great product", 5.0], ["Not worth it", 2.0], ...]
    """

    filters = [
        "sortBy=recent",
        "sortBy=helpful",
        "filterByStar=one_star",
        "filterByStar=two_star",
        "filterByStar=five_star",
    ]

    driver = build_driver()
    seen: set = set()
    all_reviews: list[Review] = []

    try:
        wait_for_login(driver, asin=asin, timeout=login_timeout)

        for params in filters:
            reviews = scrape_filter(driver, asin, params, pages, delay, seen)
            all_reviews.extend(reviews)

    finally:
        driver.quit()

    if not all_reviews:
        raise ValueError(f"No reviews found for ASIN: {asin}")

    return [[r.content, _parse_rating(r.rating)] for r in all_reviews]