import requests
from bs4 import BeautifulSoup
import pandas as pd

# Function to fetch data from a Sports Reference page
def scrape_sports_data(url, table_id):
    """
    Scrape data from a Sports Reference page.
    
    Args:
    - url (str): URL of the Sports Reference page.
    - table_id (str): ID of the HTML table containing the data.
    
    Returns:
    - pd.DataFrame: DataFrame containing the scraped data.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise error for bad status
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the table by ID
        table = soup.find('table', id=table_id)
        if table is None:
            raise ValueError(f"Table with ID '{table_id}' not found.")
        
        # Extract headers
        headers = [th.text.strip() for th in table.find('thead').find_all('th')][1:]  # Skip row headers
        
        # Extract rows
        rows = table.find('tbody').find_all('tr')
        data = []
        for row in rows:
            cells = row.find_all(['th', 'td'])
            row_data = [cell.text.strip() for cell in cells]
            data.append(row_data)
        
        # Create DataFrame
        df = pd.DataFrame(data, columns=['RowHeader'] + headers)
        return df
    except Exception as e:
        print(f"Error scraping data: {e}")
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    # Define the URL and table ID to scrape
    url = "https://www.sports-reference.com/cfb/years/2024-team-stats.html"  # Replace with the desired URL
    table_id = "team_stats"  # Replace with the actual table ID (inspect the page source)
    
    # Scrape data
    scraped_data = scrape_sports_data(url, table_id)
    
    # Save data to a CSV file
    if not scraped_data.empty:
        scraped_data.to_csv("sports_data.csv", index=False)
        print("Data scraped and saved to 'sports_data.csv'.")
    else:
        print("No data scraped.")