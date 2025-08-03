import pdfkit
import os

# Download wkhtmltopdf from https://wkhtmltopdf.org/downloads.html
# Set the path to the wkhtmltopdf executable

wkhtmltopdf_path = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"

# Configure pdfkit to use wkhtmltopdf
config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)

# Path of HTML and PDF files
path = os.getcwd()
htmlFile = f"{path}\\index.html"
pdfFile = f"{path}\\output.pdf"

# Adding PDF Options for customized view
options = {
    "page-size": "A4",
    "margin-top": "0.75in",
    "margin-right": "0.75in",
    "margin-bottom": "0.75in",
    "margin-left": "0.75in",
    "encoding": "UTF-8",
    "no-outline": None,
}

# Check if the HTML file exists before proceeding
if not os.path.exists(htmlFile):
    print(f"HTML file does not exist at: {htmlFile}")
else:
    try:
        # Convert HTML to PDF
        pdfkit.from_file(htmlFile, pdfFile, configuration=config, options=options)
        print(f"Successfully converted HTML to PDF: {pdfFile}")
    except Exception as e:
        print(f"An error occurred: {e}")
