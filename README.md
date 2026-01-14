WR_PROJECT
├───data
│   │───cleaned_data.csv                                # dataset after preprocessing
│   │───kh-polar.ver1.0.csv                             # convert kh-polar.ver1.0.text to csv file
│   └───kh-polar.ver1.0.text                            # Raw dataset from repo: "https://github.com/ye-kyaw-thu/kh-polarity?tab=readme-ov-file"                  
├───model
│   └───RandomForest_TFIDF_Smoke.pkl                    # best model using oversample smoke, tfidf, and model random forest
├───notebook
│   └───train_model
│   │   │───train_model_with_BoW_with_smoke.ipynb       # train model using BoW, Smoke, with varous model
│   │   │───train_model_with_BoW.ipynb                  # train model using BoW with varous model
│   │   │───train_model_with_CBoW_with_smoke.ipynb      # train model using CBoW, Smoke, with varous model
│   │   │───train_model_with_CBoW.ipynb                 # train model using CBoW, with varous model
│   │   │───train_model_with_TFIDF.ipynb                # train model using TFIDF, Smoke, with varous model
│   │   └───train_model_with_TFIDF_with_smoke.ipynb     # train model using TFIDF, Smoke, with varous model    
│   │───convert_text_2_csv.py                           # file convert kh-polar.ver1.0.text to kh-polar.ver1.0.csv
│   │───preprocessing_dataset.ipynb                     # file preprocessing dataset from kh-polar.ver1.0.csv to cleaned_data.csv  
│   └───stop_word_remove.ipynb                          # file clean khmer stopwords-corpus-385.csv to khmer_stopwords.csv
├───stop_word
│   │───khmer stopwords-corpus-385.csv                  # Raw stopwords from repo: "https://github.com/back-kh/KSWv2-Stop-Word-Dictionary-for-Khmer-Document"
│   └───khmer_stopwords.csv                             # Clean stopwords file
├───streamlit
│   └───app.py                                          # app for streamlit
└───vectorizers
    └───tfidf_vectorizer.pkl                            # tfidf vectorizer 
