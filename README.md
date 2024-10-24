# Python RFM Model Analysis
The project aims to analyze the retailer’s customer behavior to enhance marketing strategies for marketing teams.

## Table of Contents:
1. [Overall](#overall)
2. [Data Cleaning](#clean)
3. [RFM Segmentation](#rfm)
4. [Visualization](#vis)
5. [Insights & Recommendations](#insight)

<div id='overall'/>
  
## 1. Overall

**Platform**: Google Colab.

**Main Technique**: Exploratory data analysis (EDA), data transformations, aggregation and grouping.

**Library**: Pandas, NumPy, Seaborn, Matplotlib and Squarify.

**Context**: SuperStore, a global retail company with a large customer base, seeks to segment customers for its marketing campaigns. Marketing department has requested the Data Analysis Department's assistance in implementing the RFM model.

**RFM model:** The model segments customers based on three key metrics: Recency (how recently a purchase was made), Frequency (how often purchases occur), and Monetary (total spending amount).

**Result**: Based on the dataset, I use Python to classify customers into RFM segments, then visualize results to offer practical insights and marketing campaigns corresponding to main customer segments.
  
**Links to dataset info:** https://docs.google.com/spreadsheets/d/1yNt8-kkoDyYzq8tYbqWRqqrAfyhPNtBlfSo-9aRvbCY/view

<div id='clean'/>
  
## 2. Data Cleaning

The dataset consists of two tables: transaction information and segmentation.

- **Transaction Information**: This table has a many-to-one relationship, where each order ID is linked with multiple product IDs.
- **Segmentation**: This table includes all RFM scores for each segment.

To clean the data, we will keep records with only positive prices and units, remove canceled orders (IDs starting with 'C'), and change date formats.

### Import libraries used

```python
!pip install squarify

from google.colab import auth
auth.authenticate_user()

import gspread
from google.auth import default
creds, _ = default()
gc = gspread.authorize(creds)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
```
### Import data

```python
data_link='1yNt8-kkoDyYzq8tYbqWRqqrAfyhPNtBlfSo-9aRvbCY'
data = gc.open_by_key(data_link)

order_info = pd.DataFrame(data.worksheet('ecommerce retail').get_all_records())
segmentation = pd.DataFrame(data.worksheet('Segmentation').get_all_records())

print(order_info.head())
print('\n', segmentation.head())
```
### Prepare segmentation list

```python
segmentation.columns = ['segment','score']

segmentation['score'] = segmentation['score'].str.split(',')
segmentation = segmentation.explode('score')

segmentation['score'] = segmentation['score'].str.strip()

segmentation.info()
```
### Order info data cleaning

```python
# drop duplicate
order_info = order_info.drop_duplicates()

# drop cancel order rows
order_info = order_info[~order_info['InvoiceNo'].astype(str).str.startswith('C')]

# remove adjust bad debt order by remove row where InvoiceNo not int
order_info = order_info[order_info['InvoiceNo'].apply(lambda x: isinstance(x, int))]

# drop rows where Quantity or UnitPrice is negative or 0, or CustomerID is empty
order_info = order_info[
    (order_info['Quantity'] > 0) &
    (order_info['CustomerID'] != '') &
    (order_info['UnitPrice'] > 0)
]

# Convert InvoiceDate to datetime
order_info['InvoiceDate'] = pd.to_datetime(order_info['InvoiceDate'], format='%m/%d/%y %H:%M', errors='coerce')
# Extract date
order_info['Date'] = order_info['InvoiceDate'].dt.date
order_info['Date'] = order_info['Date'].astype('datetime64[ns]')

# Drop Invoice Date
order_info = order_info.drop(columns=['InvoiceDate'])

# Rename column
order_info = order_info.rename(columns={
    'InvoiceNo': 'invoiceid',
    'StockCode': 'stockcode',
    'Description': 'description',
    'Quantity': 'quantity',
    'UnitPrice': 'price',
    'CustomerID': 'customerid',
    'Country': 'country',
    'Date': 'date'
})

order_info.reset_index(drop=True)

order_info.info()
```

<div id='rfm'/>

## 3. RFM Segmentation

After data cleaning, I calculated the Recency (days since the last purchase), Frequency (total transactions), and Monetary value (total spending) for each customer. Then I used quintiles to assign RFM scores to these components to determine each customer's segment. Finally, we grouped by segmentation to determine the number of customers, average recency, average frequency, and total revenue for each segment.

### Create table Customer with their segment table

```python
# Create Revenue col
order_info['revenue'] = order_info['quantity'] * order_info['price']
order_info.head()

customer_rfm = order_info.groupby('customerid').agg({'date':'max',
                                         'invoiceid':'nunique',
                                         'revenue':'sum'}).reset_index()
customer_rfm['date'] = (pd.to_datetime('31/12/2011', format='%d/%m/%Y') - customer_rfm['date']).dt.days
customer_rfm.columns = ['customerid','recency','frequency','monetory']

# rank customer_rfm from 1 to 5
customer_rfm['r_rank'] = 6 - (pd.qcut(customer_rfm['recency'].rank(method='first'), 5, labels=False) + 1)
customer_rfm['f_rank'] = pd.qcut(customer_rfm['frequency'].rank(method='first'), 5, labels=False) + 1
customer_rfm['m_rank'] = pd.qcut(customer_rfm['monetory'].rank(method='first'), 5, labels=False) + 1

# customer_rfm_score
customer_rfm['score'] = customer_rfm['r_rank'].astype(str) + customer_rfm['f_rank'].astype(str) + customer_rfm['m_rank'].astype(str)

# merge with segmentation
customer_rfm = segmentation.merge(customer_rfm, on ='score', how = 'right')
customer_rfm
```
### Create RFM segment table

```python
rfm = customer_rfm.groupby('segment').agg(
    num_ctm=('customerid', 'count'),
    avg_r=('recency', 'mean'),
    avg_f=('frequency', 'mean'),
    sum_rvn=('monetory', 'sum')
).round(1).reset_index()

rfm
```

<div id='vis'/>

## 4. Data Visualization
### Prepare for visualization

```python
rfm['%_ctm'] = round((rfm['num_ctm'] / rfm['num_ctm'].sum()) * 100, 1)
rfm['%_rvn'] = round((rfm['sum_rvn'] / rfm['sum_rvn'].sum()) * 100, 1)
color = sns.color_palette("RdYlGn_r", len(rfm))
```
### Visualization for Avg Recency, Avg Frequency, and % Revenue by Segment

```python
columns = ['avg_r', 'avg_f', '%_rvn']
titles = ['Average Recency by Segment', 'Average Frequency by Segment', '% Revenue by Segment']
y_labels = ['Day(s)', 'Order Time(s)', '% Revenue']

for i, col in enumerate(columns):
    rfm = rfm.sort_values(by=col, ascending=True if i == 0 else False)
    plt.figure(figsize=(8, 5))
    plt.title(titles[i])
    bars = plt.bar(rfm['segment'], rfm[col], alpha=0.7)
    sns.barplot(x=rfm['segment'], y=rfm[col], palette=color, legend = False)
    plt.bar_label(bars, label_type="center")
    plt.xlabel('Segment')
    plt.ylabel(y_labels[i])
    plt.xticks(rotation=45)
    plt.show()
```
![Avg R](https://github.com/user-attachments/assets/ffee588b-8aa9-4ad2-871e-528c05393576)

![Avg F](https://github.com/user-attachments/assets/9f8a65b2-37fd-4356-82be-7946a4d606b7)

![% M](https://github.com/user-attachments/assets/049d2f51-511d-4758-bac3-5bcc63cfcf9e)

### Visualization for % Customer by Segment

```python
rfm = rfm.sort_values(by='%_ctm', ascending=False)
plt.figure(figsize=(16, 6))
labels_with_values = [f"{segment}\n{ctm:.1f}%" for segment, ctm in zip(rfm['segment'], rfm['%_ctm'])]
squarify.plot(sizes=rfm['%_ctm'], label=labels_with_values, color=color, alpha=0.8)
plt.title('% of Customers by Segment')
plt.axis('off')
plt.show()
```

![% Customer](https://github.com/user-attachments/assets/be7b6c87-650a-4fde-89a0-93f11a1791a1)

<div id='insight'/>

## 5. Insights and Recommendations

By the end of 2011, SuperStore had a *mixed business situation*, with strong segments like Champions, Loyal, and Potential Loyal customers, and weaker segments like Hibernating, About to Sleep, At Risk, and Lost customers. The Champions segment made up 19.2% of total customers, while Hibernating customers accounted for 15.9%.

There are clear differences between customer segments, shown by charts that highlight skewed averages for recency, frequency, and total revenue. This indicates a *significant gap* between strong and weak segments.  Despite having various segments, the company heavily relies on a few top ones, particularly the **Champions** segment, which suggests issues in: *turning new customers into loyal ones* or *customer service that may cause loyal customers to be lost*.

To enhance performance, SuperStore should focus on Recency and Frequency rather than Monetary value, as loyal shoppers often make many purchases at lower amounts throughout the year.

**Main Segments' Characteristics**
| Segment                | Characteristics                        |
|-----------------------|---------------------------------------|
| Champions             | Highly engaged and valuable customers  <br> (Score: 555, 554, 544, 545, 454, 455, 445)   |
| Loyal                 | Regularly purchases; strong retention  <br> (Score: 543, 444, 435, 355, 354, 345, 344, 335) |
| Potential Loyalist    | Engaging but not fully committed       <br> (Score: 553, 551, 552, 541, 542, 533, 532, 531, 452, 451, 442, 441, 431, 453, 433, 432, 423, 353, 352, 351, 342, 341, 333, 323) |
| Hibernating Customers  | Inactive but may return with effort    <br> (Score: 332, 322, 233, 232, 223, 222, 132, 123, 122, 212, 211) |
| About to Sleep        | Decreasing engagement; needs attention  <br> (Score: 331, 321, 312, 221, 213, 231, 241, 251) |
| At Risk               | Likely to disengage; require intervention <br> (Score: 255, 254, 245, 244, 253, 252, 243, 242, 235, 234, 225, 224, 153, 152, 145, 143, 142, 135, 134, 133, 125, 124) |
| Lost Customers        | No recent activity; may need reactivation <br> (Score: 111, 112, 121, 131, 141, 151)        |

**Main Segments' Campaign**
| Segment Type         | Campaign Name                | Description                                                                                          |
|----------------------|------------------------------|------------------------------------------------------------------------------------------------------|
| Champions            | Exclusive Rewards Program     | Launch a tiered loyalty program offering exclusive discounts, early access to sales, and gifts.     |
| Loyal                | Personalized Communication     | Send personalized emails with product recommendations based on past purchases and special dates.     |
| Potential Loyalist   | Engagement Incentives         | Provide targeted promotions and rewards to encourage repeat purchases and strengthen loyalty.         |
| Hibernating Customers | Re-Engagement Campaign        | Implement targeted email campaigns with discounts to encourage returns and highlight new offerings.   |
| About to Sleep       | Nudge Reminders               | Send reminders and special offers to re-engage customers showing signs of decreased activity.         |
| At Risk              | Win-Back Offers               | Create limited-time promotions for at-risk customers, such as discounts or exclusive offers to incentivize returns. |
| Lost Customers       | Recovery Campaign             | Develop targeted campaigns with significant discounts or free gifts to entice lost customers back.    |
