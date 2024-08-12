# Import Libraries
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np  
import mysql.connector
from mysql.connector import Error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder, StandardScaler
import locale
import logging
import calendar
import datetime
import json

# Inisialisasi Flask dan Logging
app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Konfigurasi Database
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'jurujual_pos'
}

# Variabel Global
trained_model = None
evaluation_results = None
scaler = None 
trained_stock_model = None
stock_scaler = None
trained_stock_model = None
evaluation_results_stok = None 

# Fungsi Menghapus Outlier
def remove_outliers(df, col_name):
    df_no_outliers = pd.DataFrame()
    # Group by 'Tahun' and 'Bulan' sebelum proses outliers
    grouped = df.groupby(['Tahun'])
    for (tahun), group in grouped:
        Q1 = group[col_name].quantile(0.25)
        Q3 = group[col_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Menyaring outlier untuk setiap kelompok (tahun dan bulan)
        filtered_group = group[(group[col_name] >= lower_bound) & (group[col_name] <= upper_bound)]
        df_no_outliers = pd.concat([df_no_outliers, filtered_group])
    return df_no_outliers

# Fungsi untuk menghubungkan ke database MySQL dan memuat data penjualan
def load_data():
    try:
        db_connection = mysql.connector.connect(**DB_CONFIG)
        query = """
        SELECT s.date AS Tanggal, s.reference AS Reference, s.customer_name AS Pembeli, 
            p.product_unit AS Satuan, p.product_name AS Produk, c.category_name AS Kategori,
            sd.quantity AS Qty, p.product_price AS HargaJual, sd.product_discount_amount AS Diskon,
            sp.amount AS SubTotal, s.payment_method AS MetodePembayaran, 
            s.status AS Status, s.payment_status AS StatusPembayaran
        FROM sales s
        JOIN sale_details sd ON s.id = sd.sale_id
        JOIN sale_payments sp ON s.id = sp.sale_id
        JOIN products p ON sd.product_id = p.id
        JOIN categories c ON p.category_id = c.id
        """
        df = pd.read_sql(query, con=db_connection)
        logging.info("Data berhasil dimuat dari database.")
    except Error as e:
        logging.error(f"Error saat menghubungkan ke database: {e}")
        return None
    finally:
        if db_connection.is_connected():
            db_connection.close()
            logging.info("Koneksi database ditutup.")
    # Pemrosesan data
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df['Bulan'] = df['Tanggal'].dt.month
    df['Tahun'] = df['Tanggal'].dt.year
    monthly_sales = df.groupby(['Tahun', 'Bulan']).agg({
        'SubTotal': 'sum',
        'Produk': 'nunique',
        'Reference': 'nunique',
        'Pembeli': 'nunique',
        'Satuan': 'nunique',
        'Kategori': 'nunique',
        'MetodePembayaran': 'nunique',
        'Qty': 'sum',
        'HargaJual': 'mean',
        'Diskon': 'mean'
    }).reset_index()
    monthly_sales['SubTotal'] = monthly_sales['SubTotal'] / 100
    le = LabelEncoder()
    for col in monthly_sales.columns:
        if monthly_sales[col].dtype == 'object':
            monthly_sales[col] = le.fit_transform(monthly_sales[col])
    columns_with_outliers = ['SubTotal']
    for col in columns_with_outliers:
        monthly_sales = remove_outliers(monthly_sales, col)
    return monthly_sales

# Fungsi untuk menghubungkan ke database MySQL dan memuat data stok
def load_stock_data():
    try:
        db_connection = mysql.connector.connect(**DB_CONFIG)

        query = """
        SELECT s.date AS Tanggal, s.reference AS Reference, s.customer_name AS Pembeli, 
            p.product_unit AS Satuan, p.product_name AS Produk, c.category_name AS Kategori,
            sd.quantity AS Qty, p.product_price AS HargaJual, sd.product_discount_amount AS Diskon,
            sp.amount AS SubTotal, s.payment_method AS MetodePembayaran, 
            s.status AS Status, s.payment_status AS StatusPembayaran
        
        FROM sales s
        JOIN sale_details sd ON s.id = sd.sale_id
        JOIN sale_payments sp ON s.id = sp.sale_id
        JOIN products p ON sd.product_id = p.id
        JOIN categories c ON p.category_id = c.id;
        """

        df = pd.read_sql(query, con=db_connection)
        logging.info("Data stok berhasil dimuat dari database.")
    except Error as e:
        logging.error(f"Error saat menghubungkan ke database: {e}")
        return None, None  # Return tuple of None values to handle unpacking correctly
    finally:
        if db_connection.is_connected():
            db_connection.close()
            logging.info("Koneksi database ditutup.")

    # Pemrosesan data
    df['Tanggal'] = pd.to_datetime(df['Tanggal'])
    df['Bulan'] = df['Tanggal'].dt.month
    df['Tahun'] = df['Tanggal'].dt.year

    # Menghitung penjualan bulanan untuk setiap produk
    monthly_sales_stock = df.groupby(['Tahun', 'Bulan', 'Produk']).agg({
        'SubTotal': 'sum', 
        'Reference': 'nunique',
        'Pembeli': 'nunique',
        'Satuan': 'nunique',
        'Kategori': 'nunique',
        'MetodePembayaran': 'nunique',
        'Qty': 'sum',
        'HargaJual': 'mean',
        'Diskon': 'mean'
    }).reset_index()

    # Mengubah kolom nama produk menjadi numerik
    le = LabelEncoder()
    monthly_sales_stock['Produk'] = le.fit_transform(monthly_sales_stock['Produk'])

    return monthly_sales_stock, le


# Route untuk Melakukan Training Model Penjualan
@app.route('/train-penjualan', methods=['POST'])
def train_model():
    global trained_model, evaluation_results, scaler
    # Reset hasil prediksi yang lama
    trained_model = None
    evaluation_results = None
    monthly_sales = load_data()
    if monthly_sales is None:
        return jsonify({'status': 'Gagal memuat data dari database.'}), 500
    # Menggunakan fitur yang lebih lengkap
    X = monthly_sales[['Produk','Pembeli','Kategori','Qty','HargaJual', 'Bulan','Reference']]
    y = monthly_sales['SubTotal']
    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    # Inisialisasi dan pelatihan model Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    # Evaluasi model
    y_train_pred = rf_model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = train_mse ** 0.5
    train_r2 = r2_score(y_train, y_train_pred)
    train_mape = mean_absolute_percentage_error(y_train, y_train_pred)

    y_test_pred = rf_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = test_mse ** 0.5
    test_r2 = r2_score(y_test, y_test_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred)
    # Simpan model dan hasil evaluasi
    trained_model = rf_model
    evaluation_results = {
        'Training': {
            'MSE': f"{train_mse:.2f}",
            'RMSE': f"{train_rmse:.2f}",
            'R2': f"{train_r2:.2f}",
            'MAPE': f"{train_mape:.2%}"
        },
        'Testing': {
            'MSE': f"{test_mse:.2f}",
            'RMSE': f"{test_rmse:.2f}",
            'R2': f"{test_r2:.2f}",
            'MAPE': f"{test_mape:.2%}"
        }
    }
    logging.info(f"Model berhasil dilatih: Training MSE={train_mse}, Testing MSE={test_mse}")
    # Simpan histori training ke database
    current_time = datetime.datetime.now()
    max_year = max(monthly_sales['Tahun'])
    last_two_digits_year = str(max_year)[-2:]
    training_history = {
        'training_time': current_time,
        'start_year': min(monthly_sales['Tahun']),
        'end_year': max_year,
        'model_version': f'TrainSales RFModel v1.{last_two_digits_year}',
        'training_result': evaluation_results
    }
    # Simpan ke database
    try:
        db_connection = mysql.connector.connect(**DB_CONFIG)
        cursor = db_connection.cursor()
        insert_query = """
        INSERT INTO training_histories_penjualan (training_time, start_year, end_year, model_version, training_result)
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (
            training_history['training_time'], 
            training_history['start_year'],
            training_history['end_year'],
            training_history['model_version'],
            json.dumps(training_history['training_result'])
        ))
        db_connection.commit()
        
        logging.info("Histori training berhasil disimpan.")
    except Error as e:
        logging.error(f"Error saat menyimpan histori training ke database: {e}")
    finally:
        if db_connection.is_connected():
            cursor.close()
            db_connection.close()
    return jsonify({'status': 'Model trained successfully!', 'evaluation': evaluation_results})

# Route untuk melakukan training model stok
@app.route('/train-stok', methods=['POST'])
def train_stock_model():
    global trained_stock_model, evaluation_results_stok, le
    
    # Reset hasil prediksi yang lama
    trained_stock_model = None
    evaluation_results_stok = None
    
    # Load data stok dari database
    loaded_data = load_stock_data()
    if loaded_data is None:
        return jsonify({'status': 'Gagal memuat data dari database.'}), 500

    # Unpack hasil load_data
    monthly_sales_stock, le = loaded_data
    
    # Menggunakan fitur yang lebih lengkap
    X = monthly_sales_stock[['Produk', 'Pembeli', 'Kategori', 'SubTotal', 'HargaJual', 'Bulan','Reference']]
    y = monthly_sales_stock['Qty']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Inisialisasi dan pelatihan model Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Evaluasi model
    y_train_pred = rf_model.predict(X_train)
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = train_mse ** 0.5
    train_r2 = r2_score(y_train, y_train_pred)
    train_mape = mean_absolute_percentage_error(y_train, y_train_pred)
    

    y_test_pred = rf_model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = test_mse ** 0.5
    test_r2 = r2_score(y_test, y_test_pred)
    test_mape = mean_absolute_percentage_error(y_test, y_test_pred)

    # Simpan model dan hasil evaluasi
    trained_stock_model = rf_model
    evaluation_results_stok = {
        'Training': {
            'MSE': f"{train_mse:.2f}",
            'RMSE': f"{train_rmse:.2f}",
            'R2': f"{train_r2:.2f}",
            'MAPE': f"{train_mape:.2%}"
        },
        'Testing': {
            'MSE': f"{test_mse:.2f}",
            'RMSE': f"{test_rmse:.2f}",
            'R2': f"{test_r2:.2f}",
            'MAPE': f"{test_mape:.2%}"
        }
    }

    logging.info(f"Model stok berhasil dilatih: Training MSE={train_mse}, Testing MSE={test_mse}")
    # Simpan histori training ke database
    current_time = datetime.datetime.now()
    max_year = max(monthly_sales_stock['Tahun'])
    last_two_digits_year = str(max_year)[-2:]

    training_history = {
        'training_time': current_time,
        'start_year': min(monthly_sales_stock['Tahun']),
        'end_year': max_year,
        'model_version': f'TrainStok RFModel v1.{last_two_digits_year}',
        'training_result': evaluation_results
    }

    # Simpan ke database
    try:
        db_connection = mysql.connector.connect(**DB_CONFIG)
        cursor = db_connection.cursor()

        insert_query = """
        INSERT INTO training_histories_stok (training_time, start_year, end_year, model_version, training_result)
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (
            training_history['training_time'], 
            training_history['start_year'],
            training_history['end_year'],
            training_history['model_version'],
            json.dumps(training_history['training_result'])
        ))
        db_connection.commit()
        
        logging.info("Histori training berhasil disimpan.")
    except Error as e:
        logging.error(f"Error saat menyimpan histori training ke database: {e}")
    finally:
        if db_connection.is_connected():
            cursor.close()
            db_connection.close()
    return jsonify({'status': 'Model stok berhasil dilatih!', 'evaluation': evaluation_results_stok})


@app.route('/predict-penjualan', methods=['POST'])
def predict():
    global trained_model, evaluation_results, scaler
    
    if trained_model is None or evaluation_results is None or scaler is None:
        return jsonify({'status': 'Model belum dilatih atau scaler belum diinisialisasi. Silakan lakukan training terlebih dahulu.'}), 400
    
    data = request.json
    input_years = data.get('years')

    if not input_years:
        logging.error('Invalid input data: years missing.')
        return jsonify({'status': 'Invalid input data.'}), 400

    # Load data untuk prediksi
    X = load_data()
    if X is None:
        return jsonify({'status': 'Gagal memuat data dari database untuk prediksi.'}), 500
    
    # Buat data prediksi untuk tahun-tahun yang diminta pengguna
    future_years = [int(year.strip()) for year in input_years.split(',')]

    future_data = []

    for year in future_years:
        for month in range(1, 13):  # Prediksi untuk setiap bulan dalam tahun tersebut
            future_data.append({
                'Produk': X['Produk'].mean(),
                'Pembeli': X['Pembeli'].mean(),
                'Kategori': X['Kategori'].mean(),
                'Qty': X['Qty'].mean(),
                'HargaJual': X['HargaJual'].mean(),
                'Bulan': month,
                'Reference': X['Reference'].mean(),
                'Tahun': year 
            })

    future_dates = pd.DataFrame(future_data)
    future_dates_scaled = scaler.transform(future_dates[['Produk', 'Pembeli', 'Kategori', 'Qty', 'HargaJual', 'Bulan','Reference']])
    predictions = trained_model.predict(future_dates_scaled)

    # Atur locale untuk format mata uang
    try:
        locale.setlocale(locale.LC_ALL, 'id_ID.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'id_ID')
        except locale.Error:
            locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

    formatted_predictions = [{
        'Waktu': f"{calendar.month_name[bulan]} {tahun}",
        'SubTotal Penjualan': locale.currency(pred, grouping=True, symbol=True)
    } for tahun, bulan, pred in zip(future_dates['Tahun'], future_dates['Bulan'], predictions)]

    logging.info(f"Prediksi Penjualan={formatted_predictions}")

    return jsonify({'predictions': formatted_predictions})

# Route untuk melakukan prediksi stok
@app.route('/predict-stok', methods=['POST'])
def predict_stock():
    global trained_stock_model, evaluation_results_stok, le

    loaded_data = load_stock_data()
    if loaded_data is None:
        return jsonify({'status': 'Gagal memuat data dari database.'}), 500
    monthly_sales_stock, le = loaded_data

    if trained_stock_model is None or evaluation_results_stok is None:
        return jsonify({'status': 'Model belum dilatih. Silakan lakukan training terlebih dahulu.'}), 400
    
    data = request.json
    input_choice = data.get('choice')
    input_years = data.get('years')

    logging.debug(f'Input choice: {input_choice}, Input years: {input_years}')

    if not input_choice or not input_years:
        logging.error('Invalid input data: choice or years missing.')
        return jsonify({'status': 'Invalid input data.'}), 400

    if input_choice == '1':
        future_months = [1, 2, 3, 4, 5, 6]
    elif input_choice == '2':
        future_months = [7, 8, 9, 10, 11, 12]
    else:
        return jsonify({'status': 'Pilihan tidak valid untuk choice. Masukkan 1 untuk Januari-Juni, 2 untuk Juli-Desember.'}), 400

    future_years = [int(year.strip()) for year in input_years.split(',')]

    # Ambil data dari bulan dan tahun sebelumnya yang relevan untuk prediksi
    monthly_sales_prev = monthly_sales_stock[monthly_sales_stock['Bulan'].isin(future_months)]

    if monthly_sales_prev is None or le is None:
        return jsonify({'status': 'Gagal memuat data dari database untuk prediksi stok.'}), 500

    future_data = []

    for year in future_years:
        for product in monthly_sales_prev['Produk'].unique():
            for month in future_months:
                future_data.append({
                    'Produk': product,
                    'Pembeli': monthly_sales_prev[monthly_sales_prev['Produk'] == product]['Pembeli'].mean(),
                    'Kategori': monthly_sales_prev[monthly_sales_prev['Produk'] == product]['Kategori'].mean(),
                    'SubTotal': monthly_sales_prev[monthly_sales_prev['Produk'] == product]['SubTotal'].mean(),
                    'HargaJual': monthly_sales_prev[monthly_sales_prev['Produk'] == product]['HargaJual'].mean(),
                    'Bulan': (year - monthly_sales_stock['Tahun'].max()) * 12 + month,
                    'Reference': monthly_sales_prev[(monthly_sales_prev['Produk'] == product)]['Reference'].mean()
                })

    future_df = pd.DataFrame(future_data)
    # future_scaled = scaler.transform(future_df)
    future_predictions = trained_stock_model.predict(future_df)

    future_df['PrediksiQty'] = future_predictions
    predicted_sales_future = future_df.groupby(['Produk'])['PrediksiQty'].sum().reset_index()
    predicted_sales_future['Produk'] = le.inverse_transform(predicted_sales_future['Produk'])

    predicted_sales_future['PrediksiQty'] = predicted_sales_future['PrediksiQty'].round().astype(int)
    predicted_sales_future['StokAman'] = (predicted_sales_future['PrediksiQty'] * 1.1).round().astype(int)

    result = {
        'predictions': predicted_sales_future.to_dict(orient='records')
    }

    return jsonify(result)



# Route untuk mereset hasil prediksi
@app.route('/reset-penjualan', methods=['POST'])
def reset_predictions():
    global trained_model, evaluation_results
    trained_model = None
    evaluation_results = None
    logging.info("Hasil prediksi penjualan telah di-reset.")
    return jsonify({'status': 'Hasil prediksi penjualan berhasil di-reset.'})

# Route untuk mereset hasil prediksi
@app.route('/reset-stok', methods=['POST'])
def reset_predictions_stok():
    global trained_stock_model, evaluation_results_stok
    trained_stock_model = None
    evaluation_results_stok = None
    logging.info("Hasil prediksi stok telah di-reset.")
    return jsonify({'status': 'Hasil prediksi stok berhasil di-reset.'})

if __name__ == '__main__':
    app.run(debug=True)
