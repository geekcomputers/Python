from fpdf import FPDF


# Author: @NavonilDas


class MyPdf(FPDF):
    def header(self):
        # Uncomment the line below to add logo if needed
        # self.image('somelogo.png',12,10,25,25) # Draw Image ar (12,10) with height = 25 and width = 25
        self.set_font("Arial", "B", 18)
        self.text(27, 10, "Generating PDF With python")
        self.ln(10)

    def footer(self):
        # Set Position at 1cm (10mm) From Bottom
        self.set_y(-10)
        # Arial italic 8
        self.set_font("Arial", "I", 8)
        # set Page number at the bottom
        self.cell(0, 10, "Page No {}".format(self.page_no()), 0, 0, "C")
        pass


pdf = MyPdf()
# Set Author Name of the PDF
pdf.set_author("@NavonilDas")
# Set Subject of The PDF
pdf.set_subject("python")
# Set the Title of the PDF
pdf.set_title("Generating PDF with Python")
pdf.add_page()

# Set Font family Courier with font size 28
pdf.set_font("Courier", "", 18)
# Add Text at (0,50)
pdf.text(0, 50, "Example to generate PDF in python.")

# Set Font Family Courier with italic and font size 28
pdf.set_font("Courier", "i", 28)
pdf.text(0, 60, "This is an italic text")  # Write text at 0,60

pdf.add_page()

# Center Text With border and a line break with height=10mm
pdf.cell(0, 10, "Hello There", 1, 1, "C")

# Save the Output at Local File
pdf.output("output.pdf", "F")
