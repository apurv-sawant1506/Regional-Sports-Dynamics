This script uses requests for fetching HTML and BeautifulSoup for parsing it. Replace the placeholders in the script with specific endpoints based on the data you're targeting (e.g., NFL, NBA statistics).

Steps to Use
Inspect the Site:

Visit the specific page on Sports Reference, e.g., team statistics for a specific league.
Use the browser developer tools (F12 or right-click → "Inspect") to locate the table's id attribute.
Update the Script:

Replace url with the target page's URL.
Replace table_id with the ID of the table you want to scrape.
Run the Script:

Install dependencies: pip install requests beautifulsoup4 pandas.
Execute the script. It will save the scraped data to sports_data.csv.
Caution:

Review Sports Reference’s terms of service to ensure compliance with their data scraping policies.
Use the script responsibly, avoiding excessive requests.