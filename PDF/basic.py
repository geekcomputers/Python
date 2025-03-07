from fpdf import FPDF

pdf = FPDF()
pdf.set_author("@NavonilDas")
pdf.set_subject("python")
pdf.set_title("Generating PDF with Python")
pdf.add_page()

pdf.set_font("Courier", "", 18)
pdf.text(0, 50, "Example to generate PDF in python.")

pdf.set_font("Courier", "i", 28)
pdf.text(0, 60, "This is an italic text")

pdf.rect(10, 100, 60, 30, "D")

pdf.set_fill_color(255, 0, 0)
pdf.ellipse(10, 135, 50, 50, "F")

pdf.output("output.pdf", "F")
