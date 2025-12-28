import os
import requests
import sys

# Pre-defined libraries of public domain texts
LIBRARIES = {
    "Novelist": [
        ("Moby Dick", "https://www.gutenberg.org/files/2701/2701-0.txt"),
        ("Pride and Prejudice", "https://www.gutenberg.org/files/1342/1342-0.txt"),
        ("The Great Gatsby", "https://www.gutenberg.org/cache/epub/64317/pg64317.txt"),
        ("Frankenstein", "https://www.gutenberg.org/files/84/84-0.txt"),
        ("Sherlock Holmes", "https://www.gutenberg.org/files/1661/1661-0.txt"),
    ],
    "Legal": [
        ("US Constitution", "https://www.gutenberg.org/cache/epub/5/pg5.txt"),
        ("The Federalist Papers", "https://www.gutenberg.org/files/1404/1404-0.txt"),
        ("Two Treatises of Government", "https://www.gutenberg.org/files/7370/7370-0.txt"),
        ("Leviathan", "https://www.gutenberg.org/files/3207/3207-0.txt"),
        ("International Law", "https://www.gutenberg.org/files/24675/24675-0.txt"),
    ]
}


def fetch_library(category):
    if category not in LIBRARIES:
        print(f"Error: Category '{category}' not found in library options.")
        print(f"Available: {', '.join(LIBRARIES.keys())}")
        return

    # output path: data/training/<Category>
    output_dir = os.path.join("data", "training", category)
    os.makedirs(output_dir, exist_ok=True)

    print(f"--- Fetching {category} Library ---")

    for title, url in LIBRARIES[category]:
        filename = title.replace(" ", "_").lower() + ".txt"
        save_path = os.path.join(output_dir, filename)

        if os.path.exists(save_path):
            print(f"  [Skipping] {title} (already exists)")
            continue

        print(f"  [Downloading] {title}...")
        try:
            response = requests.get(url)
            response.raise_for_status()

            # Gutenberg texts often have headers/footers.
            # Ideally we strip them, but raw text is okay for training.
            content = response.text

            with open(save_path, "w", encoding="utf-8") as f:
                f.write(content)

        except Exception as e:
            print(f"    ! Failed: {e}")

    print(f"\nDone! Saved to {output_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m tools.gutenberg_fetcher <Novelist|Legal>")
        sys.exit(1)

    fetch_library(sys.argv[1])