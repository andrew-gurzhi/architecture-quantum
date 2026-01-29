import wikipediaapi
from pathlib import Path

PAGES = [
    "Geralt of Rivia",
    "Yennefer of Vengerberg",
    "Cirilla",
    "Triss Merigold",
    "Eredin",
    "Zoltan Chivay",
    "Milva",
    "Cahir",
    "Gaunter O'Dimm",
    "Shani",
    "Philippa Eilhart",
    "Henselt",
    "Mousesack",
    "Eskel",
    "Lambert",
]

OUT_DIR = Path("data/clean")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Указываем user_agent, как требует Wikipedia API
wiki = wikipediaapi.Wikipedia(
    language="en",
    extract_format=wikipediaapi.ExtractFormat.WIKI,
    user_agent="my-python-script-for-education"
)

def extract_text(page_title: str) -> str:
    page = wiki.page(page_title)
    if not page.exists():
        raise RuntimeError(f"Page not found: {page_title}")
    text = page.text.strip()
    if not text:
        raise RuntimeError(f"No text extracted for page {page_title}")
    return text

def normalize_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def main():
    for title in PAGES:
        print(f"Downloading {title}...")
        try:
            text = extract_text(title)
        except Exception as e:
            print(f"Failed to download {title}: {e}")
            continue

        text = normalize_text(text)

        filename = OUT_DIR / f"{title.replace(' ', '_')}.txt"
        filename.write_text(text, encoding="utf-8")
        print(f"Saved {filename} ({len(text)} chars)")

if __name__ == "__main__":
    main()
