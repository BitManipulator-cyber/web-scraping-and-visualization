import subprocess
from bs4 import BeautifulSoup
import pandas as pd
import os

# Step 1: Get HTML using curl
url = "http://localhost:8080/"
command = ["curl", url]
result = subprocess.run(command, capture_output=True, text=True)
html = result.stdout

# Step 2: Parse with BeautifulSoup
soup = BeautifulSoup(html, "html.parser")

# Step 3: Find the table
table = soup.find("table")  # Finds the first <table> tag
if table:
    # Extract headers
    headers = [th.get_text(strip=True) for th in table.find_all("th")]
    
    # Extract rows
    rows = []
    for tr in table.find_all("tr")[1:]:  # Skip header row
        cells = [td.get_text(strip=True) for td in tr.find_all("td")]
        if cells:  # Avoid empty rows
            rows.append(cells)
    
    # Step 4: Convert to DataFrame
    df = pd.DataFrame(rows, columns=headers)
    print(df)

    #Saving the dataframe as a .csv file in Samples Folder
    os.makedirs("Samples", exist_ok=True)
    df.to_csv("Samples/Companies.csv", index=False)

else:
    print("No table found in HTML")
