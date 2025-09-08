import subprocess
from bs4 import BeautifulSoup
import pandas as pd
import os

# Step 1: Get HTML using curl
url = "http://localhost:8080/flight"
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
        if cells:
            rows.append(cells)
    
    # Step 4: Convert to DataFrame
    df = pd.DataFrame(rows, columns=headers)

    # --- Normalize column names (lowercase, no spaces) ---
    df.columns = df.columns.str.strip().str.lower()
    print("Extracted Columns:", df.columns.tolist())

    # --- Cleaning numeric columns ---
    df["passengers"] = df["passengers"].str.replace(",", "", regex=False).astype(int)
    df["flights"] = df["flights"].astype(int)

    df["load factor (%)"] = df["load factor (%)"].str.replace("%", "", regex=False).astype(float)

    df["revenue ($)"] = (
        df["revenue ($)"]
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(int)
    )

    def parse_growth(val):
        if val == "-" or val == "":
            return None
        return float(val.replace("%", "").replace("+", ""))

    df["passenger growth (%)"] = df["passenger growth (%)"].apply(parse_growth)

    print(df.head())  # Preview cleaned data

    # Step 5: Save to CSV inside Samples folder
    os.makedirs("Samples", exist_ok=True)
    df.to_csv("Samples/Flight.csv", index=False)

    print("Data Saved successfully to Samples/")

else:
    print("No table found in HTML")
