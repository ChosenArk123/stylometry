import wikipediaapi
import os
from pathlib import Path


def scrape_wikipedia(category_name, topics, output_dir):
    # Setup the Wikipedia API (identify your project per Wiki guidelines)
    wiki = wikipediaapi.Wikipedia(
        user_agent='StylometryProject/1.0 (contact: paruchurianish@gmail.com)',
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI
    )

    dest_folder = os.path.join(output_dir, category_name)
    os.makedirs(dest_folder, exist_ok=True)

    for topic in topics:
        print(f"Fetching: {topic}...")
        page = wiki.page(topic)

        if page.exists():
            # Clean up the filename
            safe_name = topic.replace(" ", "_").lower()
            file_path = os.path.join(dest_folder, f"{safe_name}.txt")

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(page.text)
            print(f"  -> Saved to {file_path}")
        else:
            print(f"  -> Topic '{topic}' not found.")


if __name__ == "__main__":
    # Example: Sourcing "Essayist" style data on philosophy
    philosophy_topics = ["Existentialism", "Nihilism", "Stoicism", "Epistemology"]
    scrape_wikipedia("Essayist", philosophy_topics, "data/raw_data")