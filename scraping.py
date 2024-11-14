from google_play_scraper import reviews_all, Sort, app
import pandas as pd

# set the application id for google play
app_id = 'com.ea.game.simcitymobile_row'

# scraping app reviews for indoensia country and language
print('Scraping...')
scrap_app_reviews = reviews_all(
    app_id,
    lang='id',
    country='id',
    sort=Sort.MOST_RELEVANT,
    count=100
)
print(f'\33]30m Scraping success \33]0m')

# Make a dataframe from retrieved data
app_reviews_df = pd.DataFrame(scrap_app_reviews)

# convert into csv file to donwload
csv_path = '/datasets/simcity_reviews2.csv'
print(f'Loading download csv file to {csv_path}')
app_reviews_df.to_csv(csv_path, index=False)
print(f'\33]30m Success Download csv file to {csv_path}\33]0m')

# get number of reviews and columns
num_of_reviews, num_of_columns = app_reviews_df.shape


print(f'Number of Reviews: {num_of_reviews}')
print(f'Number of Columns: {num_of_columns}')
app_reviews_df.head()
