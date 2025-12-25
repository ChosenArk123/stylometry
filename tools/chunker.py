import os
import sys
from pathlib import Path

# Add the project root to path so we can import analysis modules
sys.path.append(str(Path(__file__).parent.parent))
from analysis.preprocessing import load_text, preprocess


def chunk_file(filepath, output_dir, chunk_size=20):
    """
    Reads a file, splits it into chunks of 'chunk_size' sentences,
    and saves them as separate files.
    """
    print(f"Processing {filepath}...")
    try:
        text = load_text(filepath)
        # We use your existing preprocess to get smart sentence splitting
        doc, sentences, tokens = preprocess(text)

        filename = Path(filepath).stem

        # Group sentences into chunks
        current_chunk = []
        chunk_id = 1

        for sent in sentences:
            current_chunk.append(sent.text)

            if len(current_chunk) >= chunk_size:
                # Save chunk
                save_path = os.path.join(output_dir, f"{filename}_{chunk_id:03d}.txt")
                with open(save_path, "w", encoding="utf-8") as f:
                    f.write(" ".join(current_chunk))

                current_chunk = []
                chunk_id += 1

        # Save any leftovers
        if current_chunk:
            save_path = os.path.join(output_dir, f"{filename}_{chunk_id:03d}.txt")
            with open(save_path, "w", encoding="utf-8") as f:
                f.write(" ".join(current_chunk))

        print(f"  -> Created {chunk_id} samples.")

    except Exception as e:
        print(f"  -> Error: {e}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python -m tools.chunker <source_folder> <training_folder>")
        print("Example: python -m tools.chunker raw_data/ data/training/")
        sys.exit(1)

    source_root = sys.argv[1]
    train_root = sys.argv[2]

    # Replicate folder structure
    for author in os.listdir(source_root):
        author_path = os.path.join(source_root, author)
        if not os.path.isdir(author_path): continue

        # Create destination folder (e.g. data/training/Essayist)
        dest_path = os.path.join(train_root, author)
        os.makedirs(dest_path, exist_ok=True)

        # Chunk every file in the source author folder
        for filename in os.listdir(author_path):
            file_path = os.path.join(author_path, filename)
            chunk_file(file_path, dest_path)


if __name__ == "__main__":
    main()