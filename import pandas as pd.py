pip install openpyxl
pip install statsmodels

import pandas as pd
import numpy as np
import requests, zipfile, io

# URL of the zip file
url = "https://www.ssc.wisc.edu/~bhansen/econometrics/Econometrics%20Data.zip"

# Send a HTTP request to the URL of the zip file
r = requests.get(url)

# Create a ZipFile object from the response
z = zipfile.ZipFile(io.BytesIO(r.content))
# Extract all the contents of zip file in current directory
z.extractall()

# read the xlsx file inside the zip is named 'cps09mar.xlsx'
df = pd.read_excel('cps09mar/cps09mar.xlsx')
print(df.head())

# Import necessary libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns


# Subset of men
men = df[(df['female'] == 0) & (df['education']>12)]


men['marital'] = men['marital'].replace([1, 2, 3, 4], 'married')
men['marital'] = men['marital'].replace(5, 'divorced')
men['marital'] = men['marital'].replace(6, 'separated')
men['marital'] = men['marital'].replace(7, 'never_married')


men['marital'] = men['marital'].astype('category')

# Multinomial logistic regression
# Defining the independent and dependent variables
X = men[['age']]  # Independent variable (age)
X = sm.add_constant(X)  # Adding a constant for the intercept
y = men['marital']  # Dependent variable (marital status)

# Fitting the multinomial logit model
model = sm.MNLogit(y, X)
result = model.fit()

print(result.summary())

# Define the age range from 15 to 85
age_range = np.arange(15, 86)  # Age range for the graph (1-year steps)
X_new = pd.DataFrame({'constant':1,'age':age_range})
predicted_probs=result.predict(X_new)
print(predicted_probs)

# Predicting probabilities for each marital status as a function of age
#X_pred = sm.add_constant(pd.DataFrame({'age': age_range}))
#pred_probs = result.predict(X_pred)

# Manually set column names for the predicted probabilities based on marital statuses
#marital_statuses = ['married', 'divorced', 'separated', 'never_married']  # Excluding 'widow'

# Convert predicted probabilities to a DataFrame for plotting
#pred_probs_df = pd.DataFrame(pred_probs, columns=marital_statuses)
#pred_probs_df['age'] = age_range
#print()



# Plotting the percentages of marital status at each age
plt.figure(figsize=(10, 6))
plt.plot(age_range, predicted_probs[0], label='Divorced')
plt.plot(age_range,  predicted_probs[1], label='Married')
plt.plot(age_range,  predicted_probs[2], label='Never Married')
plt.plot(age_range,  predicted_probs[3], label='Separated')

# Adjust x-axis to show ages from 15 to 85 with ticks in steps of 10
plt.xticks(np.arange(15, 86, 10))  # X-axis with steps of 10

# Set y-axis limits to show percentage range from 0 to 100
plt.ylim(0, 1)

# Adding labels and title
plt.xlabel('Age')
plt.ylabel('Ratio of Men')
plt.title('Ratio of Marital Status by Age (Men) Using Multinomial Logit')
plt.legend(title='Marital Status')
plt.grid(True)
plt.show()

