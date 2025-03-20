from neuralNetwork import *
import pandas as pd
import os
from scaler import *
from activationFunctions import *
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import time
from scipy.stats import cumfreq
from sklearn.metrics import mean_absolute_error, precision_score, recall_score, f1_score

rejected_labels = [
    'version',
    'alive',
    'tagId',
    'success',
    'timestamp',
    'data__tagData__pressure',
    'data__anchorData',
    #'data__coordinates__x',
    #'data__coordinates__y',
    'data__coordinates__z',
    'reference__x',
    'reference__y',
    'errorCode',
    'data__tagData__gyro__x',
    'data__tagData__gyro__y',
    'data__tagData__gyro__z',
    'data__tagData__magnetic__x',
    'data__tagData__magnetic__y',
    'data__tagData__magnetic__z',
    'data__tagData__quaternion__x',
    'data__tagData__quaternion__y',
    'data__tagData__quaternion__z',
    'data__tagData__quaternion__w',
    'data__tagData__linearAcceleration__x',
    'data__tagData__linearAcceleration__y',
    'data__tagData__linearAcceleration__z',
    'data__tagData__maxLinearAcceleration',
    'data__acceleration__x',
    'data__acceleration__y',
    'data__acceleration__z',
    'data__orientation__yaw',
    'data__orientation__roll',
    'data__orientation__pitch',
    'data__metrics__latency',
    'data__metrics__rates__update',
    'data__metrics__rates__success'
]

# Ścieżka do katalogu z plikami
dir_path1 = 'pomiary/F8/'
dir_path2 = 'pomiary/F10/'
# Pusta lista do przechowywania danych z każdego pliku
dataframes = []
network, scalersX, scalersY = None, None, None

work_mode = input("\nCzy przeprowadzić uczenie modelu sieci neuronowej? [y/n]: ").lower()
if work_mode == "y":
    # Przejście przez każdy plik w katalogu
    for filename in os.listdir(dir_path1):
        if filename.endswith('.xlsx') & filename.startswith('f8_stat'):
            # Pełna ścieżka do pliku
            file_path = os.path.join(dir_path1, filename)
            #print(filename)
            # Wczytanie pliku do DataFrame
            df = pd.read_excel(file_path, index_col=[0, 1])

            # Dodanie DataFrame do listy
            dataframes.append(df)
    print(f'\nPoprawnie wczytano dane treningowe statyczne z katalogu: {dir_path1}')
    # Przejście przez każdy plik w katalogu
    for filename in os.listdir(dir_path2):
        if filename.endswith('.xlsx') & filename.startswith('f10_stat'):
            # Pełna ścieżka do pliku
            file_path = os.path.join(dir_path2, filename)
            #print(filename)
            # Wczytanie pliku do DataFrame
            df = pd.read_excel(file_path, index_col=[0, 1])

            # Dodanie DataFrame do listy
            dataframes.append(df)
    print(f'Poprawnie wczytano dane treningowe statyczne z katalogu: {dir_path2}')
    # Połączenie wszystkich DataFrame'ów w jeden
    all_data = pd.concat(dataframes, ignore_index=True)
    #print(all_data)

    # Lista do przechowywania indeksów wierszy do usunięcia
    rows_to_drop = []

    # Przejście przez każdy wiersz w danych wejściowych
    for index, row in all_data.iterrows():
        if row['success'] != 1.0:
            # Dodanie indeksu do listy
            rows_to_drop.append(index)


    # Usunięcie wierszy z danych wejściowych i wyjściowych
    all_data = all_data.drop(rows_to_drop)

    # Stworzenie nowego DataFrame z wybranymi kolumnami
    output_data = all_data[['reference__x', 'reference__y']]

    # Usunięcie niechcianych kolumn
    all_data = all_data.drop(columns=rejected_labels)

    # Przekształcenie danych do odpowiednich formatów
    X = all_data
    y = output_data

    # Przeskalowanie danych treningowych i zapisanie wartości min i max
    X_scaled, scalersX = scale_train_data(X)
    y_scaled, scalersY = scale_train_data(y)

    X_scaled = pd.DataFrame(X_scaled).values
    y_scaled = pd.DataFrame(y_scaled).values
    print(f'\nPrzeskalowano dane treningowe')
    #print("\nPrzeskalowane dane wejściowe: ")
    #print(X_scaled)
    #print("\nPrzeskalowane dane wyjściowe: ")
    #print(y_scaled)

    print("\nX_scaled contains NaN:", np.isnan(X_scaled).any())
    print("y_scaled contains NaN:", np.isnan(y_scaled).any())
    print("X_scaled contains inf:", np.isinf(X_scaled).any())
    print("y_scaled contains inf:", np.isinf(y_scaled).any())
    print("")
    # Przekształć ndarray do DataFrame
    df = pd.DataFrame(X_scaled)
    # Zapisz do pliku CSV
    df.to_csv('X_scaled.csv', index=False, header=False)

    # Definicja sieci: 2 wejścia, 2 warstwy ukryte po 64, 32 neuronów, 1 warstwa wyjściowa z 2 neuronami
    layer_sizes = [X_scaled.shape[1], 32, 64, y_scaled.shape[1]]

    # Definiowanie funkcji aktywacji dla każdej warstwy
    activation_functions = [
        [relu] * 32,
        [relu] * 64,
        [sigmoid] * y_scaled.shape[1]  # używamy linearnej funkcji aktywacji na wyjściu
    ]

    activation_derivatives = [
        [relu_derivative] * 32,
        [relu_derivative] * 64,
        [sigmoid_derivative] * y_scaled.shape[1]
    ]
    #print(all_data.columns)
    #dropout_rates = [0.0, 0.0, 0.0, 0.0, 0.0]
    # Tworzymy sieć neuronową
    network = NeuralNetwork(layer_sizes, activation_functions, activation_derivatives)

    # Przygotowanie danych treningowych w formacie (wejścia, wyjścia)
    training_data = list(zip(X_scaled, y_scaled))
    start = time.time()


    # Trenujemy sieć
    mse_history, mae_history = network.train(training_data, epochs=200, learning_rate=0.001)
    koniec = time.time()

    # Plots for MSE and MAE
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(mse_history, label='MSE')
    plt.xlabel('Epoka')
    plt.ylabel('MSE')
    plt.title('Błąd MSE dla epok')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(mae_history, label='MAE')
    plt.xlabel('Epoka')
    plt.ylabel('MAE')
    plt.title('Błąd MAE dla epok')
    plt.legend()
    plt.show()

    czas_trenowania = koniec - start
    print(f"\nCzas trenowania modelu to: {czas_trenowania} sekund")


    file_name = input("\nPodaj nazwę pliku dla modelu do zapisu: ")
    save_model(network, file_name, scalersX, scalersY)
elif work_mode == "n":
    file_name = input("\nPodaj nazwę pliku z modelem do wczytania: ")
    network, scalersX, scalersY = load_model(file_name)
else:
    raise "Błędna opcja! Wybierz poprawną opcję z zestawu [y/n]."

#Wczytywanie danych treningowych
dataframes_test = []
#"""
# Przejście przez każdy plik w katalogu
for filename in os.listdir(dir_path1):
    if  filename.endswith('.xlsx') & filename.startswith('f8_random'):
        # Pełna ścieżka do pliku
        file_path = os.path.join(dir_path1, filename)

        # Wczytanie pliku do DataFrame
        df = pd.read_excel(file_path, index_col=[0])

        # Dodanie DataFrame do listy
        dataframes_test.append(df)
print(f'Poprawnie wczytano dane testowe losowe z katalogu: {dir_path1}')
# Przejście przez każdy plik w katalogu
for filename in os.listdir(dir_path2):
    if  filename.endswith('.xlsx') & filename.startswith('f10_random'):
        # Pełna ścieżka do pliku
        file_path = os.path.join(dir_path2, filename)

        # Wczytanie pliku do DataFrame
        df = pd.read_excel(file_path, index_col=[0])

        # Dodanie DataFrame do listy
        dataframes_test.append(df)
print(f'Poprawnie wczytano dane testowe losowe z katalogu: {dir_path2}')
#"""
# Przejście przez każdy plik w katalogu
for filename in os.listdir(dir_path1):
    if filename.endswith('.xlsx') & filename.startswith('f8') & ~(filename.startswith('f8_random') | filename.startswith('f8_stat')):
        # Pełna ścieżka do pliku
        file_path = os.path.join(dir_path1, filename)
        #print(filename)
        # Wczytanie pliku do DataFrame
        df = pd.read_excel(file_path, index_col=[0, 1])

        # Dodanie DataFrame do listy
        dataframes_test.append(df)
print(f'Poprawnie wczytano dane testowe dynamiczne z katalogu: {dir_path1}')

# Przejście przez każdy plik w katalogu
for filename in os.listdir(dir_path2):
    if filename.endswith('.xlsx') & filename.startswith('f10') & ~(filename.startswith('f10_random') | filename.startswith('f10_stat')):
        # Pełna ścieżka do pliku
        file_path = os.path.join(dir_path2, filename)
        #print(filename)
        # Wczytanie pliku do DataFrame
        df = pd.read_excel(file_path, index_col=[0, 1])

        # Dodanie DataFrame do listy
        dataframes_test.append(df)
print(f'Poprawnie wczytano dane testowe dynamiczne z katalogu: {dir_path2}')

# Połączenie wszystkich DataFrame'ów w jeden
test_data = pd.concat(dataframes_test, ignore_index=True)
#print(test_data)
# Stworzenie nowego DataFrame z wybranymi kolumnami
reference_data = test_data[['reference__x', 'reference__y']]

# Usunięcie niechcianych kolumn
test_data = test_data.drop(columns=rejected_labels)

# Przekształcenie danych do odpowiednich formatów
X_test = test_data
y_test = reference_data

# Przeskalowanie danych treningowych
X_test_scaled = scale_test_data(X_test, scalersX)
y_test_scaled = scale_test_data(y_test, scalersY)

X_test_scaled = pd.DataFrame(X_test_scaled).values
y_test_scaled = pd.DataFrame(y_test_scaled).values

print(f'\nPrzeskalowano dane testowe')

#print("\nPrzeskalowane dane wejściowe: ")
#print(X_test_scaled)
#print("\nPrzeskalowane dane wyjściowe: ")
#print(y_test_scaled)

print("\nX_test_scaled contains NaN:", np.isnan(X_test_scaled).any())
print("y_test_scaled contains NaN:", np.isnan(y_test_scaled).any())
print("X_test_scaled contains inf:", np.isinf(X_test_scaled).any())
print("y_test_scaled contains inf:", np.isinf(y_test_scaled).any())
print("")

# Testowanie sieci na przeskalowanych danych testowych
y_pred_scaled = [network.feedforward(x) for x in X_test_scaled]

y_pred_scaled = pd.DataFrame(y_pred_scaled)
y_test = pd.DataFrame(y_test)
y_pred_scaled.columns = ['reference__x', 'reference__y']
y_test.columns = ['reference__x', 'reference__y']

# Odwrócenie skali przewidywań
y_pred = unscale_data(y_pred_scaled, scalersY)
actual_unscaled = y_test

# Wyświetlenie wyników
#for i in range(len(y_pred)):
#    print(f"Test case {i}: Predicted position: {y_pred.iloc[i].values}, Actual position: {actual_unscaled.iloc[i].values}")

# łączymy oba dataframe
df = pd.concat([y_pred, y_test], axis=1)

# zapisujemy do pliku csv
df.to_csv('output.csv', index=False)

# Tworzenie wykresu przewidywanych i rzeczywistych wyników
fig, axs = plt.subplots(3, 1, figsize=(12, 15))

# Wykres 1: Przewidywane wyniki
axs[0].plot(y_pred['reference__x'], y_pred['reference__y'], 'ro', markersize=1, label='Przewidywane')
axs[0].set_xlabel('Oś X')
axs[0].set_ylabel('Oś Y')
axs[0].set_title('Przewidywane wyniki')
axs[0].legend()

# Wykres 2: Rzeczywiste wyniki
axs[1].plot(y_test['reference__x'], y_test['reference__y'], 'bo', markersize=1, label='Oczekiwane')
axs[1].set_xlabel('Oś X')
axs[1].set_ylabel('Oś Y')
axs[1].set_title('Rzeczywiste wyniki')
axs[1].legend()

# Wykres 3: Pomiary
axs[2].plot(X_test['data__coordinates__x'], X_test['data__coordinates__y'], 'go', markersize=1, label='Pomiary')
axs[2].set_xlabel('Oś X')
axs[2].set_ylabel('Oś Y')
axs[2].set_title('Pomiary')
axs[2].legend()

plt.tight_layout()
plt.show()


# Obliczanie odległości euklidesowej dla przewidywań i rzeczywistych wartości względem punktów referencyjnych
distances_pred = np.sqrt((y_pred['reference__x'] - actual_unscaled['reference__x'])**2 + (y_pred['reference__y'] - actual_unscaled['reference__y'])**2)
distances_measured = np.sqrt((X_test['data__coordinates__x'] - actual_unscaled['reference__x'])**2 + (X_test['data__coordinates__y'] - actual_unscaled['reference__y'])**2)

# Tworzenie wykresów dystrybuanty dla odległości
fig, ax = plt.subplots(3, 1, figsize=(12, 15))

# Wykres dystrybuanty dla przewidywań
cumfreq_pred = cumfreq(distances_pred, numbins=10000)
ax[0].step(cumfreq_pred.lowerlimit + np.linspace(0, cumfreq_pred.binsize * cumfreq_pred.cumcount.size, cumfreq_pred.cumcount.size), cumfreq_pred.cumcount / len(distances_pred), where='mid')
ax[0].set_title('Dystrybuanta dla odległości przewidywań')
ax[0].set_xlabel('Odległość')
ax[0].set_ylabel('CDF')

# Wykres dystrybuanty dla rzeczywistych wartości
cumfreq_actual = cumfreq(distances_measured, numbins=10000)
ax[1].step(cumfreq_actual.lowerlimit + np.linspace(0, cumfreq_actual.binsize * cumfreq_actual.cumcount.size, cumfreq_actual.cumcount.size), cumfreq_actual.cumcount / len(distances_measured), where='mid')
ax[1].set_title('Dystrybuanta dla odległości zmierzonych wartości')
ax[1].set_xlabel('Odległość')
ax[1].set_ylabel('CDF')

# Wykres dystrybuanty dla obu odległości na jednym wykresie
cumfreq_pred = cumfreq(distances_pred, numbins=10000)
cumfreq_actual = cumfreq(distances_measured, numbins=10000)
ax[2].step(cumfreq_pred.lowerlimit + np.linspace(0, cumfreq_pred.binsize * cumfreq_pred.cumcount.size, cumfreq_pred.cumcount.size), cumfreq_pred.cumcount / len(distances_pred), where='mid', label='Przewidywane')
ax[2].step(cumfreq_actual.lowerlimit + np.linspace(0, cumfreq_actual.binsize * cumfreq_actual.cumcount.size, cumfreq_actual.cumcount.size), cumfreq_actual.cumcount / len(distances_measured), where='mid', label='Zmierzona')
ax[2].set_title('Dystrybuanta dla odległości przewidywań i zmierzonych wartości')
ax[2].set_xlabel('Odległość')
ax[2].set_ylabel('CDF')
ax[2].legend()

plt.tight_layout()
plt.show()

# Wypisz maksymalną wartość dla całego DataFrame
print(y_pred_scaled.max().max())

# Wypisz minimalną wartość dla całego DataFrame
print(y_pred_scaled.min().min())

print(y_pred.max().max())

# Wypisz minimalną wartość dla całego DataFrame
print(y_pred.min().min())

# Obliczanie jakości modelu
mse1 = mean_squared_error(pd.DataFrame(y_test_scaled), y_pred_scaled)
print(f"\nŚredni błąd kwadratowy (MSE) przed odskalowaniem: {mse1}")
print(len(pd.DataFrame(y_test_scaled)))
mse2 = mean_squared_error(actual_unscaled, y_pred)
print(f"\nŚredni błąd kwadratowy (MSE): {mse2}")
print(len(actual_unscaled))

# Obliczanie błędu bezwzględnego średniego (MAE)
mae = mean_absolute_error(actual_unscaled, y_pred)
print(f"\nBłąd bezwzględny średni (MAE): {mae}")


# Obliczanie odległości euklidesowej
def euclidean_distance(y_true, y_pred):
    return np.sqrt(np.sum((y_true - y_pred)**2, axis=1))

# Obliczanie naszej własnej metryki dokładności
def custom_accuracy(y_true, y_pred, threshold=100):
    distances = euclidean_distance(y_true, y_pred)
    return np.mean(distances < threshold)


# Obliczanie naszej własnej metryki dokładności
accuracy = custom_accuracy(actual_unscaled.values, X_test.values)
print(f"\nMetryka dokładności dla danych zmierzonych, procent obiektów z odległością mniejszą niż 100 od referencyjnej: {accuracy}")

# Obliczanie naszej własnej metryki dokładności
accuracy = custom_accuracy(actual_unscaled.values, y_pred.values)
print(f"\nMetryka dokładności dla danych przewidywanych, procent obiektów z odległością mniejszą niż 100 od referencyjnej: {accuracy}")