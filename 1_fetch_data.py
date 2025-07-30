# 1_fetch_data.py
import requests
import pandas as pd
import os
import time
import xml.etree.ElementTree as ET

# --- Configuration ---
YEARS_TO_QUERY = [2025, 2024, 2023, 2022]  # Years to fetch data from
DESIRED_TOTAL_RESULTS = 2000
RESULTS_PER_YEAR = 1000 # Max results to try fetching for each year
BATCH_SIZE = 100
DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "arxiv_papers.csv")
BASE_URL = "http://export.arxiv.org/api/query?"

os.makedirs(DATA_DIR, exist_ok=True)
all_paper_data = []
# The arXiv API uses the Atom XML namespace
atom_namespace = {'atom': 'http://www.w3.org/2005/Atom'}

print("Fetching papers from arXiv API by year...")
# Loop through each year to gather data
for year in YEARS_TO_QUERY:
    print(f"\n--- Querying year: {year} ---")
    # Loop through the results for the current year in batches
    for start_index in range(0, RESULTS_PER_YEAR, BATCH_SIZE):
        # Construct a query specific to the current year
        search_query = f"cat:cs.* AND submittedDate:[{year}0101 TO {year}1231]"
        params = {
            'search_query': search_query,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending',
            'start': start_index,
            'max_results': BATCH_SIZE
        }

        try:
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            root = ET.fromstring(response.text)
            entries = root.findall('atom:entry', atom_namespace)

            if not entries:
                # This is normal if a year has fewer results than RESULTS_PER_YEAR
                print(f"No more results found for {year} after index {start_index}.")
                break # Stop fetching for this year and move to the next

            for entry in entries:
                def get_text(tag_name):
                    element = entry.find(f'atom:{tag_name}', atom_namespace)
                    return element.text.strip().replace('\n', ' ') if element is not None else ''
                
                link_tag = entry.find("atom:link[@title='pdf']", atom_namespace)
                pdf_url = link_tag.get('href') if link_tag is not None else ''

                all_paper_data.append({
                    'id': get_text('id'), 'title': get_text('title'),
                    'summary': get_text('summary'), 'pdf_url': pdf_url
                })
        except Exception as e:
            print(f"\nAn error occurred during request for year {year}: {e}")

        time.sleep(1) # Be polite to the API

    # Stop fetching if we've reached our desired total
    if len(all_paper_data) >= DESIRED_TOTAL_RESULTS:
        print("\nDesired number of results reached. Stopping.")
        break

print(f"\nTotal papers processed: {len(all_paper_data)}")

# --- Save the collected results ---
if all_paper_data:
    # Trim results if we overshot the desired total
    final_data = all_paper_data[:DESIRED_TOTAL_RESULTS]
    df = pd.DataFrame(final_data)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Successfully saved {len(df)} papers to {OUTPUT_FILE}")
else:
    print("\nNo papers were fetched.")