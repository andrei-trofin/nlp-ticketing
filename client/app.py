from app_functions import load_data, show_main_page

tfidf_vec, scaler, model, token_dictionary = load_data()

show_main_page(tfidf_vec, scaler, model, token_dictionary)
