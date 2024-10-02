import pandas as pd
import numpy as np
import requests, zipfile, io

# URL of the zip file
url = "https://www.ssc.wisc.edu/~bhansen/econometrics/Econometrics%20Data.zip"

# Send a HTTP request to the URL of the zip file
r = requests.get(url)

# Create a ZipFile object from the response
z = zipfile.ZipFile(io.BytesIO(r.content))
# Extract all the contents of zip file in current directory
z.extractall()

# read the xlsx file inside the zip is named 'cps09mar.xlsx'
datos = pd.read_excel('cps09mar/cps09mar.xlsx')
print(datos.head())