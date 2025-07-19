import os

import pandas as pd
import tabula
from docx.api import Document


def process_child_directory(child_dir: str) -> dict[str, pd.DataFrame]:
    """Process a child directory and return dataframes from its files"""
    results = {}

    if not os.path.isdir(child_dir):
        print(f"Skipping {child_dir} - directory not found")
        return results

    original_dir = os.getcwd()  # Save current directory
    os.chdir(child_dir)

    # Process PDF
    pdf_name = f"Pdf1_{os.path.basename(child_dir)}.pdf"
    if os.path.isfile(pdf_name):
        try:
            results["pdf"] = tabula.read_pdf(pdf_name, pages="all")
            print(f"Read PDF: {pdf_name}")
        except Exception as e:
            print(f"Error reading PDF {pdf_name}: {e}")

    # Process DOCX
    docx_name = f"Document_{os.path.basename(child_dir)}.docx"
    if os.path.isfile(docx_name):
        try:
            doc = Document(docx_name)
            table = doc.tables[0]
            data = []
            keys = None
            for i, row in enumerate(table.rows):
                texts = [cell.text for cell in row.cells]
                if i == 0:
                    keys = tuple(texts)
                    continue
                data.append(dict(zip(keys, texts)))
            results["document"] = pd.DataFrame(data)
            print(f"Read DOCX: {docx_name}")
        except Exception as e:
            print(f"Error reading DOCX {docx_name}: {e}")

    # Process TXT
    txt_name = f"Text_{os.path.basename(child_dir)}.txt"
    if os.path.isfile(txt_name):
        try:
            results["text"] = pd.read_csv(txt_name)
            print(f"Read TXT: {txt_name}")
        except Exception as e:
            print(f"Error reading TXT {txt_name}: {e}")

    os.chdir(original_dir)  # Return to original directory
    return results


if __name__ == "__main__":
    # Get the directory where the script (main.py) is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Correct path to Parent directory (no extra nested folder)
    parent_dir = os.path.join(script_dir, "Parent")  # Fixed path

    if not os.path.isdir(parent_dir):
        print(f"Parent directory NOT found at: {parent_dir}")
        print("Please check your folder structure! It should be:")
        print(f"  {script_dir}/Parent/")
        print(f"  {script_dir}/Parent/Child1/")
        print(f"  {script_dir}/Parent/Child2/")
        print(f"  {script_dir}/Parent/Child3/")
    else:
        print(f"Processing Parent directory: {parent_dir}")

        # Process Child1, Child2, Child3
        all_data = {}
        for i in range(1, 4):
            child_name = f"Child{i}"
            child_path = os.path.join(parent_dir, child_name)
            all_data[child_name] = process_child_directory(child_path)

        # Example: Show Child1 text data if available
        if "Child1" in all_data and "text" in all_data["Child1"]:
            print("\n--- Child1 Text Data ---")
            print(all_data["Child1"]["text"].head())
