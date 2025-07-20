# Updated ai_prediction.py - Now uses the modular remove_non_finite function
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, r2_score
from data_processing import sanitized_data_df, remove_non_finite  # MODULAR IMPORT


def prepare_features(df):
    """Extract features for AI prediction"""
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['close'].rolling(window=5).std()
    df['sma_ratio'] = df['sma_5'] / df['sma_10']
    df['volume_change'] = df['volume'].pct_change()
    df['return_5d'] = df['close'].shift(-5) / df['close'] - 1

    def classify_5d_signal(r):
        if pd.isna(r):
            return np.nan
        if r > 0.03:
            return 2
        elif r < -0.03:
            return 0
        else:
            return 1

    df['signal'] = df['return_5d'].apply(classify_5d_signal)
    df['future_return'] = df['close'].shift(-1) / df['close'] - 1

    return df


def train_and_predict(processed_df):
    """Train AI models and generate predictions - NOW WITH INFINITY HANDLING"""
    try:
        df = sanitized_data_df(processed_df)
        df = prepare_features(df)

        feature_cols = ['return', 'volatility', 'sma_ratio', 'volume_change']
        df = df.dropna(subset=feature_cols + ['signal', 'future_return'])

        if len(df) < 50:
            return None, "Insufficient data for AI prediction"

        X = df[feature_cols]
        y_class = df['signal']
        y_reg = df['future_return']

        # CRITICAL FIX: Handle infinite values using modular function
        print("🔄 Checking for infinite values in features...")
        inf_before = np.isinf(X).sum().sum()
        if inf_before > 0:
            print(f"⚠️ Found {inf_before} infinite values, applying fix...")
            X = remove_non_finite(pd.DataFrame(X), feature_cols)
            print("✅ Infinite values handled successfully")
        else:
            print("✅ No infinite values found")

        # Ensure we still have data after cleaning
        if len(X) < 20:
            return None, "Insufficient clean data after removing infinite values"

        # Re-align y variables with cleaned X
        y_class = y_class.loc[X.index]
        y_reg = y_reg.loc[X.index]

        # Train-test split
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X, y_class, y_reg, test_size=0.2, random_state=42
        )

        # Train models
        print("🧠 Training RandomForest classifier...")
        clf = RandomForestClassifier(random_state=42)
        clf.fit(X_train, y_class_train)

        print("🧠 Training Ridge regressor...")
        reg = Ridge(random_state=42)
        reg.fit(X_train, y_reg_train)

        # Generate predictions
        y_pred_class = clf.predict(X_test)
        y_pred_reg = reg.predict(X_test)
        confidences = clf.predict_proba(X_test).max(axis=1)

        # Compile results
        prediction_df = pd.DataFrame({
            "date": df.loc[X_test.index]['date'].values,
            "signal": y_pred_class,
            "expected_return": y_pred_reg,
            "confidence": confidences
        })

        def generate_trade_signal(row, return_threshold=0.01, confidence_threshold=0.6):
            if row["signal"] == 2 and row["expected_return"] > return_threshold and row[
                "confidence"] > confidence_threshold:
                return "strong_buy"
            elif row["signal"] == 0 and row["expected_return"] < -return_threshold and row[
                "confidence"] > confidence_threshold:
                return "strong_sell"
            elif row["signal"] == 1:
                return "weak_hold"
            else:
                return "ignore"

        prediction_df["final_signal"] = prediction_df.apply(generate_trade_signal, axis=1)

        signal_map = {0: "sell", 1: "hold", 2: "buy"}
        prediction_df['signal_label'] = prediction_df['signal'].map(signal_map)

        # Calculate metrics
        accuracy = accuracy_score(y_class_test, y_pred_class)
        mae = mean_absolute_error(y_reg_test, y_pred_reg)
        mse = mean_squared_error(y_reg_test, y_pred_reg)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_reg_test, y_pred_reg)

        # Calculate percentage error for visualization
        def compute_regression_percent_error(y_true, y_pred):
            actual = pd.Series(y_true).replace(0, np.nan)
            predicted = pd.Series(y_pred)
            error = ((predicted - actual) / actual) * 100
            return error.replace([np.inf, -np.inf], np.nan).dropna()

        reg_percent_error = compute_regression_percent_error(y_reg_test.values, y_pred_reg)

        print(f"✅ AI Prediction completed successfully!")
        print(f"📊 Classification Accuracy: {accuracy:.1%}")
        print(f"📊 Regression R²: {r2:.4f}")
        print(f"📊 Predictions generated: {len(prediction_df)}")

        return {
            'prediction_df': prediction_df,
            'test_data': {
                'y_class_test': y_class_test,
                'y_pred_class': y_pred_class,
                'y_reg_test': y_reg_test,
                'y_pred_reg': y_pred_reg,
                'confidences': confidences,
                'reg_percent_error': reg_percent_error
            },
            'metrics': {
                'accuracy': accuracy,
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'r2': r2
            },
            'signal_map': signal_map,
            'feature_importance': dict(zip(feature_cols, clf.feature_importances_))
        }, "Success"

    except Exception as e:
        print(f"❌ Error in train_and_predict: {str(e)}")
        return None, f"Error: {str(e)}"


def quick_prediction_test(processed_df):
    """
    NEW: Quick test function to validate data before full analysis
    """
    try:
        df = sanitized_data_df(processed_df)
        df = prepare_features(df)

        feature_cols = ['return', 'volatility', 'sma_ratio', 'volume_change']
        test_data = df[feature_cols].dropna()

        # Check for infinite values
        inf_count = np.isinf(test_data).sum().sum()
        nan_count = test_data.isna().sum().sum()

        return {
            'status': 'ready' if inf_count == 0 and len(test_data) >= 50 else 'issues',
            'data_points': len(test_data),
            'infinite_values': inf_count,
            'nan_values': nan_count,
            'message': f"Data ready for AI analysis" if inf_count == 0 and len(test_data) >= 50
            else f"Issues: {inf_count} inf values, {nan_count} NaN values, {len(test_data)} data points"
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }


if __name__ == "__main__":
    # Test the updated ai_prediction module
    print("Testing updated ai_prediction.py with infinity handling...")

    # This would be called with actual data in real usage
    print("✅ Module loaded successfully with modular infinity handling")
    print("✅ Functions available: train_and_predict, quick_prediction_test")
    print("✅ Now properly imports remove_non_finite from data_processing")