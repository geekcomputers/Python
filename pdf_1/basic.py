from fpdf import FPDF

# Author: @NavonilDas


pdf = FPDF()
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

# Draw a Rectangle at (10,100) with Width 60,30
pdf.rect(10, 100, 60, 30, "D")

# Set Fill color
pdf.set_fill_color(255, 0, 0)  # Red = (255,0,0)

# Draw a Circle at (10,135) with diameter 50
pdf.ellipse(10, 135, 50, 50, "F")

# Save the Output at Local File
pdf.output("output.pdf", "F")
