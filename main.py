import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
from xgboost import XGBRegressor
import lightgbm as lgb
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')
# Set a professional plot style
plt.style.use('seaborn-v0_8-whitegrid')

class EnhancedStockPredictor:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.close_prices = None
        self.scaler = MinMaxScaler()
        self.best_models = {}
        self.predictions = {}
        self.metrics = {}

    def load_and_prepare_data(self):
        """
        Load data, handle dates or reverse order, and create technical indicators.
        """
        print("📊 Loading and preparing data...")
        self.data = pd.read_csv(self.file_path)

        if 'Date' in self.data.columns:
            self.data['Date'] = pd.to_datetime(self.data['Date'])
            self.data = self.data.sort_values('Date').reset_index(drop=True)
        else:
            self.data = self.data.iloc[::-1].reset_index(drop=True)

        # Create a clean, non-NaN version for original analysis
        self.original_close_prices = self.data['Close'].dropna().reset_index(drop=True)

        # Create advanced features
        self.create_technical_indicators()

    def create_technical_indicators(self):
        """
        Create a rich set of technical indicators to be used as features.
        """
        print("🔧 Engineering advanced features...")
        self.data['MA_5'] = self.data['Close'].rolling(window=5).mean()
        self.data['MA_10'] = self.data['Close'].rolling(window=10).mean()
        self.data['MA_20'] = self.data['Close'].rolling(window=20).mean()
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        # Bollinger Bands
        self.data['BB_Middle'] = self.data['Close'].rolling(window=20).mean()
        bb_std = self.data['Close'].rolling(window=20).std()
        self.data['BB_Upper'] = self.data['BB_Middle'] + (bb_std * 2)
        self.data['BB_Lower'] = self.data['BB_Middle'] - (bb_std * 2)
        self.data['Returns'] = self.data['Close'].pct_change()
        self.data['Volatility'] = self.data['Returns'].rolling(window=10).std()
        self.data['Momentum'] = self.data['Close'] - self.data['Close'].shift(10)

        self.data.dropna(inplace=True)
        self.close_prices_advanced = self.data['Close']

    # --- START: METHODS FOR ORIGINAL ANALYSIS PLOTS ---
    def run_original_analysis(self):
        """
        Runs the entire original, simpler analysis and generates all its plots.
        """
        print("\n" + "="*80)
        print("🚀 STARTING ORIGINAL ANALYSIS & PLOTTING")
        print("="*80)

        # 1. Original Historical Plot
        plt.figure(figsize=(12, 6))
        plt.plot(self.original_close_prices)
        plt.xlabel('Time (Days)')
        plt.ylabel('Close Prices (USD)')
        plt.title('Original Historical Plot: AAPL Stock Close Prices')
        plt.grid(True)
        plt.show()

        # 2. Train and Plot Original Traditional Models
        self.train_and_plot_original_traditional_models()

        # 3. Train and Plot Original LSTM
        self.train_and_plot_original_lstm()

    def train_and_plot_original_traditional_models(self):
        print("\n🤖 Training original simple ML models...")
        X = np.arange(len(self.original_close_prices)).reshape(-1, 1)
        y = self.original_close_prices

        models_config = {
            'Support Vector Machines (SVM)': (SVR(), {'C': [1, 10], 'gamma': [0.01, 0.1]}),
            'Random Forest': (RandomForestRegressor(), {'n_estimators': [100, 200], 'max_depth': [5, 10]}),
            'XGBoost': (XGBRegressor(), {'learning_rate': [0.1, 0.01], 'max_depth': [3, 5]}),
            'LightGBM': (LGBMRegressor(verbose=-1), {'learning_rate': [0.1, 0.01], 'max_depth': [3, 5]})
        }

        for name, (model, params) in models_config.items():
            print(f"  Tuning {name}...")
            grid_search = GridSearchCV(model, params, scoring='neg_mean_squared_error', cv=3)
            grid_search.fit(X, y)
            predictions = grid_search.best_estimator_.predict(X)
            rmse = np.sqrt(mean_squared_error(y, predictions))

            # Original Model Performance Plot
            plt.figure(figsize=(12, 6))
            plt.plot(y, label='Actual Historical Price', color='blue')
            plt.plot(predictions, label='Predicted Historical Price', color='orange', linestyle='--')
            plt.title(f"Original Model Performance: {name} (RMSE: {rmse:.2f})")
            plt.xlabel('Time (Days)')
            plt.ylabel('Stock Price (USD)')
            plt.legend()
            plt.grid(True)
            plt.show()

    def train_and_plot_original_lstm(self):
        print("\n🧠 Training original simple LSTM model...")
        data_reshaped = self.original_close_prices.values.reshape(-1, 1)
        scaler_orig = MinMaxScaler()
        data_normalized = scaler_orig.fit_transform(data_reshaped)

        train_size = int(len(data_normalized) * 0.8)
        train_data = data_normalized[:train_size]

        model = Sequential([
            LSTM(50, activation='relu', input_shape=(1, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(train_data[:-1].reshape(-1, 1, 1), train_data[1:], epochs=50, batch_size=32, verbose=0)

        all_preds_scaled = model.predict(data_normalized[:-1].reshape(-1, 1, 1))
        all_preds = scaler_orig.inverse_transform(all_preds_scaled)

        # Original LSTM Historical Fit Plot
        plt.figure(figsize=(12, 6))
        plt.plot(self.original_close_prices[1:], label='Actual Historical Price', color='blue')
        plt.plot(all_preds, label='LSTM Predicted Price', color='red', linestyle='--')
        plt.title('Original LSTM Historical Fit')
        plt.xlabel('Time (Days)')
        plt.ylabel('Stock Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.show()
    # --- END: METHODS FOR ORIGINAL ANALYSIS PLOTS ---

    def run_advanced_analysis(self):
        """
        Runs the entire advanced analysis pipeline.
        """
        print("\n" + "="*80)
        print("🚀 STARTING ADVANCED ANALYSIS & PLOTTING")
        print("="*80)
        self.plot_technical_analysis()
        self.prepare_data_for_advanced_models()
        self.train_advanced_lstm()
        self.train_advanced_traditional_models()
        self.create_performance_dashboard()
        self.predict_future_with_confidence()

    def plot_technical_analysis(self):
        print("📈 Plotting advanced technical analysis...")
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle('Advanced Technical Analysis', fontsize=16, fontweight='bold')
        axes[0, 0].plot(self.data.index, self.data['Close'], label='Close Price', linewidth=2)
        axes[0, 0].plot(self.data.index, self.data['MA_20'], label='20-Day MA', alpha=0.7)
        axes[0, 0].set_title('Price with Moving Averages'); axes[0, 0].legend()
        axes[0, 1].plot(self.data.index, self.data['Close'], label='Close Price', linewidth=2)
        axes[0, 1].plot(self.data.index, self.data['BB_Upper'], label='Upper Band', alpha=0.5)
        axes[0, 1].plot(self.data.index, self.data['BB_Lower'], label='Lower Band', alpha=0.5)
        axes[0, 1].fill_between(self.data.index, self.data['BB_Upper'], self.data['BB_Lower'], alpha=0.1)
        axes[0, 1].set_title('Bollinger Bands'); axes[0, 1].legend()
        axes[1, 0].plot(self.data.index, self.data['RSI'], color='purple', linewidth=2)
        axes[1, 0].axhline(70, color='r', linestyle='--', label='Overbought (70)')
        axes[1, 0].axhline(30, color='g', linestyle='--', label='Oversold (30)')
        axes[1, 0].set_title('Relative Strength Index (RSI)'); axes[1, 0].legend()
        axes[1, 1].plot(self.data.index, self.data['Volatility'], color='red', linewidth=2)
        axes[1, 1].set_title('10-Day Price Volatility')
        plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()


    def prepare_data_for_advanced_models(self, sequence_length=10):
        feature_columns = ['Close', 'MA_5', 'MA_10', 'RSI', 'Volatility', 'Momentum']
        self.feature_data = self.data[[col for col in feature_columns if col in self.data.columns]]
        self.normalized_data = self.scaler.fit_transform(self.feature_data)
        X_lstm, y_lstm = [], []
        for i in range(sequence_length, len(self.normalized_data)):
            X_lstm.append(self.normalized_data[i-sequence_length:i])
            y_lstm.append(self.normalized_data[i, 0])
        self.X_lstm, self.y_lstm = np.array(X_lstm), np.array(y_lstm)

    def train_advanced_lstm(self):
        print("🧠 Training Advanced LSTM Model...")
        train_size = int(len(self.X_lstm) * 0.8)
        X_train, X_test = self.X_lstm[:train_size], self.X_lstm[train_size:]
        y_train, y_test = self.y_lstm[:train_size], self.y_lstm[train_size:]
        model = Sequential([LSTM(100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])), Dropout(0.2), LSTM(50), Dropout(0.2), Dense(25), Dense(1)])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=0)
        self.best_models['Advanced LSTM'] = model
        y_pred = model.predict(X_test)
        dummy = np.zeros((len(y_pred), self.normalized_data.shape[1])); dummy[:, 0] = y_pred.flatten()
        y_pred_orig = self.scaler.inverse_transform(dummy)[:, 0]
        y_test_orig = self.scaler.inverse_transform(np.pad(y_test.reshape(-1, 1), ((0, 0), (0, self.normalized_data.shape[1]-1))))[:, 0]
        self.calculate_advanced_metrics(y_test_orig, y_pred_orig, 'Advanced LSTM')


    def train_advanced_traditional_models(self):
        print("🤖 Training Advanced Traditional ML Models...")
        X = self.feature_data.drop('Close', axis=1)
        y = self.feature_data['Close']
        tscv = TimeSeriesSplit(n_splits=5)
        models_config = {'Random Forest': (RandomForestRegressor(random_state=42), {'n_estimators': [100, 200], 'max_depth': [10, 20]}),'XGBoost': (XGBRegressor(random_state=42), {'learning_rate': [0.1, 0.01], 'max_depth': [3, 5]}),'LightGBM': (LGBMRegressor(random_state=42, verbose=-1), {'learning_rate': [0.1, 0.01], 'num_leaves': [31, 50]})}
        for name, (model, params) in models_config.items():
            print(f"  Training {name}...")
            gs = GridSearchCV(model, params, scoring='neg_root_mean_squared_error', cv=tscv, n_jobs=-1)
            gs.fit(X, y)
            self.best_models[name] = gs.best_estimator_
            self.calculate_advanced_metrics(y, gs.predict(X), name)


    def calculate_advanced_metrics(self, y_true, y_pred, model_name):
        self.metrics[model_name] = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R²': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
            'Directional Accuracy': np.mean((np.diff(y_true) > 0) == (np.diff(y_pred) > 0)) * 100
        }

    def create_performance_dashboard(self):
        print("📊 Creating Performance Dashboard...")
        metrics_df = pd.DataFrame(self.metrics).T
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Model Performance Dashboard', fontsize=16, fontweight='bold')
        metrics_to_plot = ['RMSE', 'MAE', 'R²', 'MAPE', 'Directional Accuracy']
        for i, metric in enumerate(metrics_to_plot):
            ax = axes.flatten()[i]
            sns.barplot(x=metrics_df.index, y=metrics_df[metric], ax=ax, palette='viridis')
            ax.set_title(f'{metric} Comparison'); ax.tick_params(axis='x', rotation=45)
        if 'Random Forest' in self.best_models:
            rf = self.best_models['Random Forest']
            importances = pd.Series(rf.feature_importances_, index=self.feature_data.drop('Close', axis=1).columns)
            importances.nlargest(10).sort_values().plot(kind='barh', ax=axes.flatten()[5])
            axes.flatten()[5].set_title('Top 10 Feature Importances (RF)')
        plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.show()
        print("\n📈 Detailed Performance Metrics:"); print(metrics_df.round(4))

    def predict_future_with_confidence(self, days=10, sequence_length=10):
        print(f"🔮 Forecasting next {days} days with confidence intervals...")
        model = self.best_models['Advanced LSTM']
        mc_predictions = []
        last_sequence = self.normalized_data[-sequence_length:].reshape(1, sequence_length, self.normalized_data.shape[1])
        for _ in range(50):
            future_preds_scaled = []
            current_sequence = last_sequence.copy()
            for _ in range(days):
                pred = model(current_sequence, training=True)
                future_preds_scaled.append(pred[0, 0])
                new_row = current_sequence[0, -1, :].copy(); new_row[0] = pred
                new_row = new_row.reshape(1, 1, self.normalized_data.shape[1])
                current_sequence = np.append(current_sequence[:, 1:, :], new_row, axis=1)
            mc_predictions.append(future_preds_scaled)
        mc_predictions = np.array(mc_predictions)
        dummy = np.zeros((mc_predictions.shape[0] * days, self.normalized_data.shape[1])); dummy[:, 0] = mc_predictions.flatten()
        mc_predictions_orig = self.scaler.inverse_transform(dummy)[:, 0].reshape(mc_predictions.shape)
        mean_preds = np.mean(mc_predictions_orig, axis=0); std_devs = np.std(mc_predictions_orig, axis=0)
        upper_ci = mean_preds + 1.96 * std_devs; lower_ci = mean_preds - 1.96 * std_devs
        self.plot_future_forecast(mean_preds, upper_ci, lower_ci, days)


    def plot_future_forecast(self, predictions, upper_ci, lower_ci, days):
        plt.figure(figsize=(14, 8))
        last_day_index = self.close_prices_advanced.index[-1]
        future_days_index = np.arange(last_day_index + 1, last_day_index + 1 + days)
        plt.plot(self.close_prices_advanced.index, self.close_prices_advanced, label='Historical Prices', color='blue')
        plt.plot(future_days_index, predictions, label='Predicted Prices', color='red', linestyle='--')
        plt.fill_between(future_days_index, lower_ci, upper_ci, alpha=0.2, color='red', label='95% CI')
        plt.title(f'AAPL Stock Price: Historical Data and {days}-Day Forecast'); plt.xlabel('Time (Days)'); plt.ylabel('Stock Price ($)'); plt.legend(); plt.show()
        print(f"\n🎯 Stock Price Predictions for Next {days} Days:")
        for i, (pred, lower, upper) in enumerate(zip(predictions, lower_ci, upper_ci), 1):
            print(f"Day {i:2d}: ${pred:7.2f} (95% CI: ${lower:7.2f} - ${upper:7.2f})")


# --- Main Execution Flow ---
if __name__ == "__main__":
    predictor = EnhancedStockPredictor('AAPL_short_volume.csv')

    # Load data and create all features
    predictor.load_and_prepare_data()

    # Run the original, simpler analysis first
    predictor.run_original_analysis()

    # Run the new, advanced analysis
    predictor.run_advanced_analysis()

    print("\n✅ Complete analysis finished!")