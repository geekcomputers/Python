from xlwt import Workbook
import openpyxl

# Workbook is created
wb = Workbook()

# add_sheet is used to create sheet.
sheet1 = wb.add_sheet('Sheet 1')

sheet1.write(1, 0, 'ISBT DEHRADUN')
sheet1.write(2, 0, 'SHASTRADHARA')
sheet1.write(3, 0, 'CLEMEN TOWN')
sheet1.write(4, 0, 'RAJPUR ROAD')
sheet1.write(5, 0, 'CLOCK TOWER')
sheet1.write(0, 1, 'ISBT DEHRADUN')
sheet1.write(0, 2, 'SHASTRADHARA')
sheet1.write(0, 3, 'CLEMEN TOWN')
sheet1.write(0, 4, 'RAJPUR ROAD')
sheet1.write(0, 5, 'CLOCK TOWER')

wb.save('xlwt example.xls')

# Workbook is created
openpyxl_wb = openpyxl.Workbook()

# create_sheet is used to create sheet.
sheet1 = openpyxl_wb.create_sheet("Sheet 1")

sheet1.cell(1, 1, 'ISBT DEHRADUN')
sheet1.cell(2, 1, 'SHASTRADHARA')
sheet1.cell(3, 1, 'CLEMEN TOWN')
sheet1.cell(4, 1, 'RAJPUR ROAD')
sheet1.cell(5, 1, 'CLOCK TOWER')
sheet1.cell(1, 1, 'ISBT DEHRADUN')
sheet1.cell(1, 2, 'SHASTRADHARA')
sheet1.cell(1, 3, 'CLEMEN TOWN')
sheet1.cell(1, 4, 'RAJPUR ROAD')
sheet1.cell(1, 5, 'CLOCK TOWER')

openpyxl_wb.save("openpyxl example.xlsx")
