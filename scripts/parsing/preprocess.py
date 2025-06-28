import os
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered

def process_pdfs(input_dir, output_dir):
    """
    Convert all PDFs in a directory to Markdown using Marker and save them to the output directory.
    
    Args:
        input_dir (str): Directory containing PDF files.
        output_dir (str): Directory where Markdown files will be saved.
    """
    # Ensures that the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initializes the PdfConverter with default settings
    converter = PdfConverter(
        artifact_dict=create_model_dict(),
    )

    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".md")
            print(f"Processing: {pdf_path}")

            try:
                # Converts the PDF to rendered Markdown
                rendered = converter(pdf_path)
                markdown, _, _ = text_from_rendered(rendered)

                # Saves the Markdown to a file
                with open(output_path, "w", encoding="utf-8") as md_file:
                    md_file.write(markdown)
                
                print(f"Saved Markdown to: {output_path}")
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")

if __name__ == "__main__":
    input_directory = "data/raw"
    output_directory = "data/interim/megaset/Markdown"

    process_pdfs(input_directory, output_directory)


