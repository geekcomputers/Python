# pip install pdf2docx
# Import the required modules
from pdf2docx import Converter


def convert_pdf_to_docx(pdf_file_path, docx_file_path):
    """
    Converts a PDF file to a DOCX file using pdf2docx library.

    Parameters:
    - pdf_file_path (str): The path to the input PDF file.
    - docx_file_path (str): The desired path for the output DOCX file.

    Returns:
    None
    """
    # Convert PDF to DOCX using pdf2docx library

    # Using the built-in function, convert the PDF file to a document file by saving it in a variable.
    cv = Converter(pdf_file_path)

    # Storing the Document in the variable's initialised path
    cv.convert(docx_file_path)

    # Conversion closure through the function close()
    cv.close()


# Example usage

# Keeping the PDF's location in a separate variable
# pdf_file_path = r"D:\coding\CODE_WAR\blogs\python_tuts\book_on_python.pdf"
# # Maintaining the Document's path in a separate variable
# docx_file_path = r"D:\coding\CODE_WAR\blogs\python_tuts\book_on_python_edit.docx"

# Keeping the PDF's location in a separate variable
pdf_file_path = (
    r"C:\Users\playn\OneDrive\Desktop\read_kar_ke_feedback_le_aur_del_kar_de.pdf"
)
# Maintaining the Document's path in a separate variable
docx_file_path = (
    r"C:\Users\playn\OneDrive\Desktop\read_kar_ke_feedback_le_aur_del_kar_de.docx"
)

# Call the function to convert PDF to DOCX
convert_pdf_to_docx(pdf_file_path, docx_file_path)

# # Error handling
# # IF present then ask for permission else continue


# import fitz
# from docx import Document
# import pytesseract
# from PIL import Image


# class PDFToDocxConverter:
#     """
#     A class to convert PDF to DOCX with OCR using PyMuPDF, pytesseract, and python-docx.
#     """

#     def __init__(self, pdf_path, docx_path):
#         """
#         Initializes the PDFToDocxConverter.

#         Parameters:
#         - pdf_path (str): The path to the input PDF file.
#         - docx_path (str): The desired path for the output DOCX file.
#         """
#         self.pdf_path = pdf_path
#         self.docx_path = docx_path

#     def convert_pdf_to_docx(self):
#         """
#         Converts the PDF to DOCX with OCR and saves the result.
#         """
#         doc = Document()

#         with fitz.open(self.pdf_path) as pdf:
#             for page_num in range(pdf.page_count):
#                 page = pdf[page_num]
#                 image_list = page.get_images(full=True)

#                 for img_index, img_info in enumerate(image_list):
#                     img = page.get_pixmap(image_index=img_index)
#                     img_path = f"temp_image_{img_index}.png"
#                     img.writePNG(img_path)

#                     text = pytesseract.image_to_string(Image.open(img_path))
#                     doc.add_paragraph(text)

#         doc.save(self.docx_path)


# if __name__ == "__main__":
#     # Example usage
#     # Keeping the PDF's location in a separate variable
#     pdf_file_path = r"D:\coding\CODE_WAR\blogs\python_tuts\book_on_python.pdf"
#     # Maintaining the Document's path in a separate variable
#     docx_file_path = r"D:\coding\CODE_WAR\blogs\python_tuts\book_on_python_edit.docx"

#     converter = PDFToDocxConverter(pdf_file_path, docx_file_path)
# #     converter.convert_pdf_to_docx()


# # failed experiment.
