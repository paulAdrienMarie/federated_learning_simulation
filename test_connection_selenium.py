from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

def test_connection(url):
    
    options = Options()
    options.add_argument('--headless')  # Run in headless mode for no UI
    service = Service('/usr/local/bin/geckodriver')  # Update this path to your WebDriver
    driver = webdriver.Firefox(service=service, options=options)
    
    try:
        # Open the web application
        driver.get(url)
        
        # Wait until the page is loaded and a specific element is present (modify the selector as needed)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, 'body'))  # You can change this to a more specific element
        )
        
        # Check if the page contains expected content (modify this as needed)
        page_title = driver.title
        if page_title:
            print(f"Success: Connected to the web app. Page title is '{page_title}'.")
        else:
            print("Error: Unable to retrieve the page title.")
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        # Clean up and close the browser
        driver.quit()

if __name__ == "__main__":
    # Replace with the URL where your web app is running
    for i in range(3):
        web_app_url = 'http://localhost:8080/federated'  # Update this URL if needed
        test_connection(web_app_url)
