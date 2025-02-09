import numpy as np

table_data = {
  "columns": [
    "2016",
    "2015",
    "2017",
    "2013",
    "2014"
  ],
  "data": [
    [
      "7280",
      "7231",
      "7664",
      "7025",
      "7166"
    ],
    [
      "884",
      "792",
      "891",
      "527",
      "675"
    ],
    [
      "553",
      "693",
      "466",
      "125",
      "608"
    ],
    [
      "12.50",
      "8.61",
      "???",
      "5.34",
      "7.07"
    ],
    [
      "1316",
      "1311",
      "1317",
      "1715",
      "1609"
    ],
    [
      "2.16",
      "1.75",
      "2.60",
      "0.52",
      "1.03"
    ],
    [
      "6543",
      None,
      "6565",
      "6376",
      "6426"
    ],
    [
      "590",
      "416",
      "493",
      "269",
      "348"
    ],
    [
      "3322",
      "3375",
      "3322",
      "3375",
      "3669"
    ],
    [
      "2014",
      "77",
      "2014",
      "2014",
      "48"
    ],
    [
      "847",
      "887",
      "838",
      "268",
      "778"
    ],
    [
      "12.61",
      "8.68",
      "10.79",
      "5.41",
      "7.14"
    ]
  ],
  "Queries": [
    "Can you calculate the total sales and service revenues from 2013 to 2017",
    "how this figure compares to the total operating income over those years"
  ],
  "index": [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11
  ]
}

table_df = pd.DataFrame(table_data['data'], columns=table_data['columns'])

# Sub-query 1: Calculate the total sales and service revenues from 2013 to 2017
sales_service_revenues = table_df[['2013', '2014', '2015', '2016', '2017']].apply(pd.to_numeric, errors='coerce').sum().sum()
print(round(sales_service_revenues, 2))

# Sub-query 2: Compare this figure to the total operating income over those years
operating_income = table_df[['2013', '2014', '2015', '2016', '2017']].apply(pd.to_numeric, errors='coerce').sum().sum()
difference = sales_service_revenues - operating_income
print(round(difference, 2))