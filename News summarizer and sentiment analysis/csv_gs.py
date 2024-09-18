#-------------------------------------------------------------------------------
# Imports
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By  # Updated locator method
import time

#-------------------------------------------------------------------------------
# Setup

# Open the CSV file
with open('indian_news_with_summaries_final.csv', 'r', encoding = 'utf-8') as csv_file:

    csv_reader = csv.reader(csv_file)

    next(csv_reader)

    #-------------------------------------------------------------------------------
    # Web Automation

    # Initialize the Chrome driver (opens the browser once)
    driver = webdriver.Chrome()

    # Open the Google Form page
    driver.get('https://docs.google.com/forms/d/e/1FAIpQLSe4VWyesxkTpHi2inirYbiMCQUG5YiVNP9hDRZhAlt4NdzUhg/viewform')

    # Loop through each row in the CSV
    for line in csv_reader:

        # Wait for 2 seconds for the form to load
        time.sleep(2)

        # Fill the name field
        name_field = driver.find_element(By.XPATH, '//*[@id="mG61Hd"]/div[2]/div/div[2]/div[1]/div/div/div[2]/div/div[1]/div/div[1]/input')
        name_field.send_keys(line[0])

        # Fill the age field
        newspaper_field = driver.find_element(By.XPATH, '//*[@id="mG61Hd"]/div[2]/div/div[2]/div[2]/div/div/div[2]/div/div[1]/div/div[1]/input')
        newspaper_field.send_keys(line[1])

        # Fill the score field
        date_field = driver.find_element(By.XPATH, '//*[@id="mG61Hd"]/div[2]/div/div[2]/div[3]/div/div/div[2]/div/div/div[2]/div[1]/div/div[1]/input')
        date_field.send_keys(line[2])

        # Fill the URL field
        url_field = driver.find_element(By.XPATH, '//*[@id="mG61Hd"]/div[2]/div/div[2]/div[4]/div/div/div[2]/div/div[1]/div/div[1]/input')
        url_field.send_keys(line[3])

        # Fill the Headline field
        headline_field = driver.find_element(By.XPATH, '//*[@id="mG61Hd"]/div[2]/div/div[2]/div[5]/div/div/div[2]/div/div[1]/div/div[1]/input')
        headline_field.send_keys(line[4])

        # Fill the Content field
        content_field = driver.find_element(By.XPATH, '//*[@id="mG61Hd"]/div[2]/div/div[2]/div[6]/div/div/div[2]/div/div[1]/div[2]/textarea')
        content_field.send_keys(line[5])

        # Fill the Summary field
        summary_field = driver.find_element(By.XPATH, '//*[@id="mG61Hd"]/div[2]/div/div[2]/div[7]/div/div/div[2]/div/div[1]/div[2]/textarea')
        summary_field.send_keys(line[7])
        
        # Fill the Category field
        category_field = driver.find_element(By.XPATH, '//*[@id="mG61Hd"]/div[2]/div/div[2]/div[8]/div/div/div[2]/div/div/span/div/div[22]/div/span/div/div/div[1]/input')
        category_field.send_keys(line[6])

        # Click the submit button
        submit = driver.find_element(By.XPATH, '//*[@id="mG61Hd"]/div[2]/div/div[3]/div[1]/div[1]/div/span/span')
        submit.click()

        # Wait for submission to complete and page to load
        time.sleep(2)

        # After submission, click "Submit another response"
        another_response = driver.find_element(By.LINK_TEXT, 'Submit another response')
        another_response.click()

        # Wait for the new form to load
        time.sleep(2)

    # Close the browser after processing all rows
    driver.quit()

#-------------------------------------------------------------------------------
