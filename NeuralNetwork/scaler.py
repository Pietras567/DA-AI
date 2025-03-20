import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

def scale_train_data(train_df):
    scaler = MinMaxScaler(feature_range=(0, 1))  # Tworzenie instancji skalera
    #scaler = StandardScaler()
    scaler.fit(train_df)  # Dopasowanie skalera do danych treningowych
    scaled_train_df = pd.DataFrame(scaler.transform(train_df), columns=train_df.columns)
    return scaled_train_df, scaler

def scale_test_data(test_df, scaler):
    # Użycie tego samego skalera do przeskalowania danych testowych
    scaled_test_df = pd.DataFrame(scaler.transform(test_df), columns=test_df.columns)
    return scaled_test_df

def unscale_data(scaled_df, scaler):
    # Odwrócenie skalowania
    unscaled_df = pd.DataFrame(scaler.inverse_transform(scaled_df), columns=scaled_df.columns)
    return unscaled_df