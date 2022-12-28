import pandas as pd
import matplotlib
import scipy
import seaborn
import random 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import asyncio


# Pyodide is a port of CPython to WebAssembly/Emscripten.
# used for download
from pyodide.http import pyfetch
import requests
"""
output = 'output'
try: 
    os.mkdir(output) 
except OSError as error: 
    print(error)  
"""
URL ='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv' 

filename ="Cust_Segmentation.csv"

async def download(url, filename):
    response = await pyfetch(url)
    if response.status == 200:
        with open(filename, "wb") as f: # write binary mode
            f.write(await response.bytes())

asyncio.run(download(URL, "Cust_Segmentation.csv"))
