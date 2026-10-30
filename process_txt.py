import os
import re

def clean_brsr_text(raw_text: str) -> str:
    """
    Clean ESG/BRSR raw text for extraction.
    Steps:
      1. Remove headers, footers, page numbers.
      2. Remove table of contents or index-like lines.
      3. Normalize whitespace.
      4. Keep only clean text.
    """

    text = raw_text

    # ------------------------------
    # 1. Remove headers/footers patterns
    # ------------------------------
    # Common patterns like "Page X of Y"
    text = re.sub(r"Page\s*\d+\s*(of\s*\d+)?", " ", text, flags=re.IGNORECASE)

    # Remove lines with just page numbers
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

    # Remove company repeated headers (assume lines in ALL CAPS with length < 50)
    text = re.sub(r"(?m)^[A-Z\s]{5,50}$", " ", text)

    # ------------------------------
    # 2. Remove table of contents or index-like blocks
    # ------------------------------
    # Lines with dot leaders: "1. Introduction ........ 5"
    text = re.sub(r"\.{2,}\s*\d+", " ", text)

    # Remove TOC-style multi-line if present (lots of dots in a line)
    text = re.sub(r"(?m)^.*\.{5,}.*$", " ", text)

    # ------------------------------
    # 3. Normalize spaces and newlines
    # ------------------------------
    # Convert multiple newlines into one
    text = re.sub(r"\n{2,}", "\n", text)

    # Remove duplicate spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Trim each line
    text = "\n".join(line.strip() for line in text.splitlines())

    # Remove empty lines at start/end
    text = text.strip()

    # ------------------------------
    # 4. Lower noise
    # ------------------------------
    # Remove long sequences of underscores/dashes
    text = re.sub(r"[_\-]{5,}", " ", text)

    return text


def process_text(raw_path, cleaned_folder="cleaned_input_txt"):
    """
    Read raw text file, clean it, and save cleaned output.
    """
    os.makedirs(cleaned_folder, exist_ok=True)

    with open(raw_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    cleaned_text = clean_brsr_text(raw_text)

    company = os.path.splitext(os.path.basename(raw_path))[0]
    cleaned_path = os.path.join(cleaned_folder, f"{company}.txt")

    with open(cleaned_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"✨ Cleaned text saved at {cleaned_path}")
    return cleaned_path


# ------------------------------
# Example usage
# ------------------------------
if __name__ == "__main__":
    process_text("raw_input_txt/CIPLA_2024-25_ESG.txt")
