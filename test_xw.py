# import xlwings as xw

# wb = xw.Book()  # this will create a new workbook
# wb = xw.Book('FileName.xlsx')  # connect to an existing file in the current working directory
# wb = xw.Book(r'C:\Users\tiger\Documents\Plotus_project\test_data.xlsx')  # on Windows: use raw strings to escape backslashes
# xw.apps[0].books['FileName.xlsx'] # if several inctances of excel

#sht = wb.sheets['Sheet1'] # instantiating current excel sheet

# cell2 = xw.Range('A2') # fast access to active sheet and active workbook, not for .py scripts for some reason

# sht.range('A1').value = 'Foo 1' # set value to cell
# cell1 = sht.range('A1').value # read value from cell

# sht.range('A1').value = [['Foo 1', 'Foo 2', 'Foo 3'], [10.0, 20.0, 30.0]] # value expanding as I undrstnd is creating dictionary and assign values for keys taken from cells
# sht.range('A1').expand().value

# import matplotlib.pyplot as plt
# fig = plt.figure()
# plt.plot([1, 2, 3, 4, 5])
# [<matplotlib.lines.Line2D at 0x1071706a0>]
# sht.pictures.add(fig, name='MyPlot', update=True)
# hello.py
import numpy as np
import xlwings as xw

def some():
    wb = xw.Book.caller()
    wb.sheets[0].range('A1').value = 'Hello World!'

# print(cell1);
# print(cell2);
