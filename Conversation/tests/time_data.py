from datetime import datetime 
import time 

timestamp = time.time()
timestamp = time.time()

print(timestamp)
date_time = datetime.fromtimestamp(timestamp)
str_date_time = date_time.strftime("%d-%m-%Y, %H:%M:%S")
print("Current timestamp", str_date_time)