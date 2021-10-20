# extract number of rows using Python
import xlrd

# Give the location of the file
loc = ("sample.xlsx")
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)
sheet.cell_value(0, 0)
# Extracting number of rows
print(sheet.nrows)

# extract number of columns in Python
print(sheet.ncols)

# extracting all columns name in Python
for i in range(sheet.ncols):
    print(sheet.cell_value(0, i))

# extracting first column
sheet = wb.sheet_by_index(0)
for i in range(sheet.nrows):
    print(sheet.cell_value(i, 0))

# extract a particular row value
sheet = wb.sheet_by_index(0)
print(sheet.row_values(1))
