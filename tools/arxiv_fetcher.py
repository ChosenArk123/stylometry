import urllib.request
import feedparser
import os
import sys
from pathlib import Path


def fetch_arxiv_abstracts(query, category_name, max_results=20):
    """
    Queries the arXiv API for papers matching the query and saves their
    abstracts as text files for stylometric training.
    """
    print(f"Searching arXiv for: '{query}'...")

    # Encode the query for a URL
    encoded_query = urllib.parse.quote(query)
    base_url = 'http://export.arxiv.org/api/query?'
    search_query = f'search_query=all:{encoded_query}&start=0&max_results={max_results}'

    try:
        # Perform the request
        response = urllib.request.urlopen(base_url + search_query).read()
        feed = feedparser.parse(response)

        if not feed.entries:
            print("  -> No papers found for that query.")
            return

        # Setup destination: data/raw_data/<CategoryName>
        dest_folder = os.path.join("data", "raw_data", category_name)
        os.makedirs(dest_folder, exist_ok=True)

        count = 0
        for entry in feed.entries:
            # Create a safe filename from the title (first 30 chars)
            safe_title = "".join([c for c in entry.title[:30] if c.isalnum() or c == ' ']).rstrip()
            safe_title = safe_title.replace(" ", "_").lower()
            file_path = os.path.join(dest_folder, f"arxiv_{safe_title}.txt")

            # We save the Title + Summary to create a substantial block of academic text
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"{entry.title}\n\n{entry.summary}")

            count += 1
            print(f"  [{count}] Saved: {entry.title[:50]}...")

        print(f"\nSuccessfully fetched {count} academic samples to {dest_folder}")

    except Exception as e:
        print(f"  -> Error fetching from arXiv: {e}")


def main():
    if len(sys.argv) < 3:
        print("Usage: python -m tools.arxiv_fetcher <query> <category_name> [max_results]")
        print("Example: python -m tools.arxiv_fetcher 'quantum computing' Academic 25")
        sys.exit(1)

    query = sys.argv[1]
    category = sys.argv[2]

    # Default to 20 results if not specified
    max_res = 20
    if len(sys.argv) > 3:
        try:
            max_res = int(sys.argv[3])
        except ValueError:
            print("Error: max_results must be an integer.")
            sys.exit(1)

    fetch_arxiv_abstracts(query, category, max_res)


if __name__ == "__main__":
    main()