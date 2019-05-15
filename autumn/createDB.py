from sqlalchemy import create_engine
import pandas as pd
import glob
engine = create_engine('sqlite:///autumn.db', echo=False)

csvfileList = glob.glob('xls/*.csv')

for filename in csvfileList:
    dfname = filename.split('\\')[1].split('.')[0]
    df = pd.read_csv(filename)
    df.to_sql(dfname, con=engine)


excelFileList = glob.glob('xls/*.xlsx')

for filename in excelFileList:
    xls = pd.ExcelFile(filename)
    # Now you can list all sheets in the file

    if(len(xls.sheet_names) == 1):
      df_name = xls.sheet_names[0]
      print(df_name)
    else:
      numSheets = 0
      while numSheets < len(xls.sheet_names):
          #print(xls.sheet_names[numSheets])
          df_name = xls.sheet_names[numSheets]
          print(filename.split('\\')[1].split('.')[0] + '_' + df_name)
          numSheets = numSheets + 1

#res = engine.execute("SELECT * FROM metadata").fetchall()

#print(res)