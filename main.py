from download_data import data_read
from fake_news import check_null_title, check_null_text, check_null_label, list_columns, clean_column, predict_model, count_fake


if __name__ == '__main__':
    data_read()
    check_null_title()
    check_null_text()
    check_null_label()
    list_columns()
    clean_column()
    predict_model()
    count_fake()
