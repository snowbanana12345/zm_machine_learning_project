import src.data_base_module.data_blocks as data

test_date = data.Date(day = 12, month = 5, year = 2017)
print(test_date)
print(test_date.get_day())
print(test_date.get_month())
print(test_date.get_year())
print(test_date.get_str())

