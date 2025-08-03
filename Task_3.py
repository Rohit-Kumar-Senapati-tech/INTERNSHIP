# Task 3: Complete End-to-End Data Science Project - Fixed Version

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# Flask imports
from flask import Flask, request, jsonify, render_template_string
import json

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

class HousePriceDataScience:
    """
    Complete Data Science Pipeline for House Price Prediction
    """
    
    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.model = None
        self.feature_names = None
        
    def generate_synthetic_data(self, n_samples=5000):
        """
        Step 1: Data Collection - Generate synthetic house price data
        """
        print("Step 1: Generating synthetic house price dataset...")
        
        np.random.seed(42)
        
        # Generate features
        data = {
            'bedrooms': np.random.randint(1, 6, n_samples),
            'bathrooms': np.random.randint(1, 4, n_samples),
            'sqft_living': np.random.normal(2000, 800, n_samples),
            'sqft_lot': np.random.normal(7500, 3000, n_samples),
            'floors': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples, p=[0.3, 0.1, 0.4, 0.1, 0.1]),
            'waterfront': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'view': np.random.randint(0, 5, n_samples),
            'condition': np.random.randint(1, 6, n_samples),
            'grade': np.random.randint(3, 14, n_samples),
            'age': np.random.randint(0, 100, n_samples),
            'renovated': np.random.choice([0, 1], n_samples, p=[0.8, 0.2]),
            'zipcode': np.random.choice(range(98001, 98200), n_samples),
            'lat': np.random.normal(47.5, 0.3, n_samples),
            'long': np.random.normal(-122.3, 0.3, n_samples)
        }
        
        # Ensure positive values for area features
        data['sqft_living'] = np.abs(data['sqft_living'])
        data['sqft_lot'] = np.abs(data['sqft_lot'])
        
        # Create price based on realistic relationships
        price = (
            data['bedrooms'] * 15000 +
            data['bathrooms'] * 10000 +
            data['sqft_living'] * 150 +
            data['sqft_lot'] * 2 +
            data['floors'] * 5000 +
            data['waterfront'] * 200000 +
            data['view'] * 10000 +
            data['condition'] * 8000 +
            data['grade'] * 12000 +
            (100 - data['age']) * 1000 +
            data['renovated'] * 25000 +
            np.random.normal(0, 50000, n_samples)  # Add noise
        )
        
        # Ensure positive prices
        price = np.maximum(price, 50000)
        data['price'] = price
        
        self.data = pd.DataFrame(data)
        
        print(f"Generated dataset with {len(self.data)} samples and {len(self.data.columns)} features")
        print("\nSample data:")
        print(self.data.head())
        print(f"\nPrice statistics:")
        print(f"Mean: ${self.data['price'].mean():,.2f}")
        print(f"Median: ${self.data['price'].median():,.2f}")
        print(f"Min: ${self.data['price'].min():,.2f}")
        print(f"Max: ${self.data['price'].max():,.2f}")
        
        return self.data
    
    def exploratory_data_analysis(self):
        """
        Step 2: Exploratory Data Analysis and Visualization
        """
        print("\nStep 2: Performing Exploratory Data Analysis...")
        
        # Basic statistics
        print("Dataset Info:")
        print(f"Shape: {self.data.shape}")
        print(f"Missing values: {self.data.isnull().sum().sum()}")
        print(f"Duplicate rows: {self.data.duplicated().sum()}")
        
        # Create comprehensive visualizations
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Price distribution
        plt.subplot(3, 3, 1)
        plt.hist(self.data['price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Price Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Price ($)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 2. Price vs Square Footage
        plt.subplot(3, 3, 2)
        plt.scatter(self.data['sqft_living'], self.data['price'], alpha=0.6, s=20)
        plt.title('Price vs Living Area', fontsize=14, fontweight='bold')
        plt.xlabel('Square Feet Living', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 3. Price by bedrooms
        plt.subplot(3, 3, 3)
        bedroom_data = [self.data[self.data['bedrooms']==i]['price'] for i in range(1, 6)]
        plt.boxplot(bedroom_data, labels=[f'{i} BR' for i in range(1, 6)])
        plt.title('Price by Number of Bedrooms', fontsize=14, fontweight='bold')
        plt.xlabel('Bedrooms', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 4. Correlation heatmap
        plt.subplot(3, 3, 4)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        corr_matrix = self.data[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdYlBu_r', center=0, 
                   square=True, fmt='.2f', cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        # 5. Price by waterfront
        plt.subplot(3, 3, 5)
        waterfront_data = [self.data[self.data['waterfront']==0]['price'], 
                          self.data[self.data['waterfront']==1]['price']]
        plt.boxplot(waterfront_data, labels=['No Waterfront', 'Waterfront'])
        plt.title('Price by Waterfront', fontsize=14, fontweight='bold')
        plt.ylabel('Price ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 6. Feature importance preview (using correlation)
        plt.subplot(3, 3, 6)
        price_corr = self.data.corr()['price'].abs().sort_values(ascending=False)[1:11]
        bars = plt.bar(range(len(price_corr)), price_corr.values, color='lightcoral')
        plt.title('Top 10 Features Correlation with Price', fontsize=14, fontweight='bold')
        plt.ylabel('Absolute Correlation', fontsize=12)
        plt.xticks(range(len(price_corr)), price_corr.index, rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, price_corr.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 7. Price by grade
        plt.subplot(3, 3, 7)
        grade_price = self.data.groupby('grade')['price'].mean()
        plt.plot(grade_price.index, grade_price.values, marker='o', linewidth=2, markersize=8)
        plt.title('Average Price by Grade', fontsize=14, fontweight='bold')
        plt.xlabel('Grade', fontsize=12)
        plt.ylabel('Average Price ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 8. Age vs Price
        plt.subplot(3, 3, 8)
        plt.scatter(self.data['age'], self.data['price'], alpha=0.6, s=20)
        plt.title('Price vs House Age', fontsize=14, fontweight='bold')
        plt.xlabel('Age (years)', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 9. Price distribution by condition
        plt.subplot(3, 3, 9)
        condition_data = [self.data[self.data['condition']==i]['price'] for i in range(1, 6)]
        plt.boxplot(condition_data, labels=[f'Cond {i}' for i in range(1, 6)])
        plt.title('Price by Condition', fontsize=14, fontweight='bold')
        plt.xlabel('Condition Rating', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=3.0)
        plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Additional summary statistics
        print("\nKey Insights:")
        print(f"- Waterfront premium: ${self.data[self.data['waterfront']==1]['price'].mean() - self.data[self.data['waterfront']==0]['price'].mean():,.0f}")
        print(f"- Price per sqft range: ${(self.data['price']/self.data['sqft_living']).min():.0f} - ${(self.data['price']/self.data['sqft_living']).max():.0f}")
        print(f"- Most common bedrooms: {self.data['bedrooms'].mode().iloc[0]}")
        print(f"- Most common grade: {self.data['grade'].mode().iloc[0]}")
        
        print("EDA visualizations saved as 'eda_analysis.png'")
        
    def preprocess_data(self):
        """
        Step 3: Data Preprocessing
        """
        print("\nStep 3: Preprocessing data...")
        
        # Create additional features
        print("Creating engineered features...")
        self.data['price_per_sqft'] = self.data['price'] / self.data['sqft_living']
        self.data['total_rooms'] = self.data['bedrooms'] + self.data['bathrooms']
        self.data['is_luxury'] = (self.data['grade'] >= 10).astype(int)
        self.data['lot_ratio'] = self.data['sqft_lot'] / self.data['sqft_living']
        self.data['renovated_old'] = ((self.data['renovated'] == 1) & (self.data['age'] > 30)).astype(int)
        
        print(f"Added engineered features: price_per_sqft, total_rooms, is_luxury, lot_ratio, renovated_old")
        
        # Handle outliers (using IQR method for price)
        print("Removing outliers...")
        Q1 = self.data['price'].quantile(0.25)
        Q3 = self.data['price'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        initial_size = len(self.data)
        self.data = self.data[(self.data['price'] >= lower_bound) & 
                             (self.data['price'] <= upper_bound)]
        print(f"Removed {initial_size - len(self.data)} outliers ({((initial_size - len(self.data))/initial_size)*100:.1f}%)")
        
        # Prepare features and target
        feature_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                       'waterfront', 'view', 'condition', 'grade', 'age', 'renovated',
                       'lat', 'long', 'total_rooms', 'is_luxury', 'lot_ratio', 'renovated_old']
        
        X = self.data[feature_cols]
        y = self.data['price']
        
        # Split data with stratification based on price quartiles
        print("Splitting data...")
        price_quartiles = pd.qcut(y, q=4, labels=False)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=price_quartiles
        )
        
        # Scale features
        print("Scaling features...")
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        self.feature_names = feature_cols
        
        print(f"Training set: {self.X_train.shape[0]} samples")
        print(f"Test set: {self.X_test.shape[0]} samples")
        print(f"Features: {len(feature_cols)}")
        print(f"Feature names: {feature_cols}")
        
        return self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test
    
    def train_models(self):
        """
        Step 4: Model Training and Selection
        """
        print("\nStep 4: Training and comparing models...")
        
        # Define models with better parameters
        models = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=200, 
                max_depth=15, 
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=150, 
                learning_rate=0.1, 
                max_depth=6,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        }
        
        results = {}
        
        # Train and evaluate each model
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            if name == 'Linear Regression':
                model.fit(self.X_train_scaled, self.y_train)
                y_pred_train = model.predict(self.X_train_scaled)
                y_pred = model.predict(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred_train = model.predict(self.X_train)
                y_pred = model.predict(self.X_test)
            
            # Evaluate
            mse = mean_squared_error(self.y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(self.y_test, y_pred)
            r2 = r2_score(self.y_test, y_pred)
            
            # Training metrics
            train_r2 = r2_score(self.y_train, y_pred_train)
            
            results[name] = {
                'model': model,
                'predictions': y_pred,
                'train_predictions': y_pred_train,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'train_r2': train_r2,
                'overfit': train_r2 - r2
            }
            
            print(f"Training R²: {train_r2:.4f}")
            print(f"Test R²: {r2:.4f}")
            print(f"RMSE: ${rmse:,.2f}")
            print(f"MAE: ${mae:,.2f}")
            print(f"Overfitting: {train_r2 - r2:.4f}")
        
        # Select best model (highest test R²)
        best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
        self.model = results[best_model_name]['model']
        
        print(f"\nBest model: {best_model_name}")
        print(f"Best Test R²: {results[best_model_name]['r2']:.4f}")
        print(f"Best RMSE: ${results[best_model_name]['rmse']:,.2f}")
        
        # Feature importance for tree-based models
        if hasattr(self.model, 'feature_importances_'):
            print(f"\nTop 10 Feature Importances ({best_model_name}):")
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
                print(f"{i+1:2d}. {row['feature']:15s}: {row['importance']:.4f}")
        
        # Visualize results
        self.visualize_model_results(results)
        
        return results, best_model_name
    
    def visualize_model_results(self, results):
        """Visualize model performance with improved clarity"""
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Model comparison - R² scores
        plt.subplot(3, 3, 1)
        model_names = list(results.keys())
        train_r2_scores = [results[name]['train_r2'] for name in model_names]
        test_r2_scores = [results[name]['r2'] for name in model_names]
        
        x_pos = np.arange(len(model_names))
        width = 0.35
        
        bars1 = plt.bar(x_pos - width/2, train_r2_scores, width, label='Training R²', alpha=0.8)
        bars2 = plt.bar(x_pos + width/2, test_r2_scores, width, label='Test R²', alpha=0.8)
        
        plt.title('Model R² Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('R² Score', fontsize=12)
        plt.xticks(x_pos, model_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 2. Model comparison - RMSE
        plt.subplot(3, 3, 2)
        rmse_scores = [results[name]['rmse'] for name in model_names]
        bars = plt.bar(model_names, rmse_scores, color=['lightblue', 'lightgreen', 'lightcoral'], alpha=0.8)
        plt.title('Model RMSE Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('RMSE ($)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, rmse_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(rmse_scores)*0.01,
                    f'${value:,.0f}', ha='center', va='bottom', fontsize=10)
        
        # 3. Best model predictions vs actual
        best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
        best_predictions = results[best_model_name]['predictions']
        
        plt.subplot(3, 3, 3)
        plt.scatter(self.y_test, best_predictions, alpha=0.6, s=30)
        
        # Perfect prediction line
        min_val = min(self.y_test.min(), best_predictions.min())
        max_val = max(self.y_test.max(), best_predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.title(f'{best_model_name}: Predicted vs Actual', fontsize=14, fontweight='bold')
        plt.xlabel('Actual Price ($)', fontsize=12)
        plt.ylabel('Predicted Price ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add R² annotation
        r2_score_val = results[best_model_name]['r2']
        plt.text(0.05, 0.95, f'R² = {r2_score_val:.4f}', transform=plt.gca().transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=12)
        
        # 4. Residuals plot
        plt.subplot(3, 3, 4)
        residuals = self.y_test - best_predictions
        plt.scatter(best_predictions, residuals, alpha=0.6, s=30)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        plt.title(f'{best_model_name}: Residuals Plot', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Price ($)', fontsize=12)
        plt.ylabel('Residuals ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 5. Residuals histogram
        plt.subplot(3, 3, 5)
        plt.hist(residuals, bins=30, alpha=0.7, color='lightblue', edgecolor='black')
        plt.title('Residuals Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Residuals ($)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        plt.axvline(residuals.mean(), color='red', linestyle='--', label=f'Mean: ${residuals.mean():,.0f}')
        plt.axvline(residuals.std(), color='orange', linestyle='--', label=f'Std: ${residuals.std():,.0f}')
        plt.legend()
        
        # 6. Feature importance (if available)
        if hasattr(results[best_model_name]['model'], 'feature_importances_'):
            plt.subplot(3, 3, 6)
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': results[best_model_name]['model'].feature_importances_
            }).sort_values('importance', ascending=True).tail(10)
            
            plt.barh(range(len(feature_importance)), feature_importance['importance'])
            plt.yticks(range(len(feature_importance)), feature_importance['feature'])
            plt.title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
            plt.xlabel('Importance', fontsize=12)
            plt.grid(True, alpha=0.3)
        
        # 7. Model performance metrics comparison
        plt.subplot(3, 3, 7)
        metrics = ['R²', 'RMSE', 'MAE']
        metric_values = {
            name: [results[name]['r2'], results[name]['rmse']/1000, results[name]['mae']/1000] 
            for name in model_names
        }
        
        x_pos = np.arange(len(metrics))
        width = 0.25
        
        for i, (name, values) in enumerate(metric_values.items()):
            plt.bar(x_pos + i*width, values, width, label=name, alpha=0.8)
        
        plt.title('Model Metrics Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Score / Value (RMSE & MAE in $1000s)', fontsize=12)
        plt.xticks(x_pos + width, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 8. Prediction error by price range
        plt.subplot(3, 3, 8)
        price_ranges = pd.qcut(self.y_test, q=5, labels=['Low', 'Med-Low', 'Medium', 'Med-High', 'High'])
        error_by_range = []
        range_labels = []
        
        for range_label in ['Low', 'Med-Low', 'Medium', 'Med-High', 'High']:
            mask = price_ranges == range_label
            if mask.any():
                errors = np.abs(self.y_test[mask] - best_predictions[mask])
                error_by_range.append(errors)
                range_labels.append(range_label)
        
        plt.boxplot(error_by_range, labels=range_labels)
        plt.title('Prediction Error by Price Range', fontsize=14, fontweight='bold')
        plt.xlabel('Price Range', fontsize=12)
        plt.ylabel('Absolute Error ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # 9. Learning curve simulation
        plt.subplot(3, 3, 9)
        train_sizes = np.linspace(0.1, 1.0, 10)
        train_scores = []
        test_scores = []
        
        for size in train_sizes:
            sample_size = int(len(self.X_train) * size)
            if sample_size < 10:
                continue
                
            if best_model_name == 'Linear Regression':
                temp_model = LinearRegression()
                temp_model.fit(self.X_train_scaled[:sample_size], self.y_train[:sample_size])
                train_pred = temp_model.predict(self.X_train_scaled[:sample_size])
                test_pred = temp_model.predict(self.X_test_scaled)
            else:
                temp_model = type(results[best_model_name]['model'])(**results[best_model_name]['model'].get_params())
                temp_model.fit(self.X_train[:sample_size], self.y_train[:sample_size])
                train_pred = temp_model.predict(self.X_train[:sample_size])
                test_pred = temp_model.predict(self.X_test)
            
            train_scores.append(r2_score(self.y_train[:sample_size], train_pred))
            test_scores.append(r2_score(self.y_test, test_pred))
        
        train_sizes_actual = train_sizes[:len(train_scores)]
        plt.plot(train_sizes_actual, train_scores, 'o-', label='Training Score', linewidth=2)
        plt.plot(train_sizes_actual, test_scores, 'o-', label='Test Score', linewidth=2)
        plt.title('Learning Curve', fontsize=14, fontweight='bold')
        plt.xlabel('Training Set Size (fraction)', fontsize=12)
        plt.ylabel('R² Score', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout(pad=3.0)
        plt.savefig('model_results.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        print("Model results visualizations saved as 'model_results.png'")
    
    def save_model(self):
        """Save the trained model and preprocessing objects"""
        print("\nStep 5: Saving model and preprocessing objects...")
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        
        # Save model and scaler
        joblib.dump(self.model, 'models/house_price_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        # Save feature names and model info
        model_info = {
            'feature_names': self.feature_names,
            'model_type': type(self.model).__name__,
            'n_features': len(self.feature_names),
            'training_date': pd.Timestamp.now().isoformat()
        }
        
        with open('models/model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print("Model saved successfully!")
        print("Files saved:")
        print("- models/house_price_model.pkl")
        print("- models/scaler.pkl")
        print("- models/model_info.json")
    
    def run_complete_pipeline(self):
        """Run the complete data science pipeline"""
        print("="*60)
        print("COMPLETE DATA SCIENCE PROJECT - HOUSE PRICE PREDICTION")
        print("="*60)
        
        try:
            # Execute all steps
            self.generate_synthetic_data()
            self.exploratory_data_analysis()
            self.preprocess_data()
            results, best_model = self.train_models()
            self.save_model()
            
            print("\n" + "="*60)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Best Model: {best_model}")
            print(f"Best R²: {results[best_model]['r2']:.4f}")
            print(f"Best RMSE: ${results[best_model]['rmse']:,.2f}")
            print(f"Training R²: {results[best_model]['train_r2']:.4f}")
            print(f"Overfitting: {results[best_model]['overfit']:.4f}")
            print("Ready for deployment!")
            
            return results
            
        except Exception as e:
            print(f"Pipeline failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

# Flask Web Application for Model Deployment
class HousePriceFlaskApp:
    """
    Flask web application for house price prediction
    """
    
    def __init__(self):
        self.app = Flask(__name__)
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.model_info = None
        self.load_model()
        self.setup_routes()
    
    def load_model(self):
        """Load the trained model and preprocessing objects"""
        try:
            self.model = joblib.load('models/house_price_model.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            
            with open('models/model_info.json', 'r') as f:
                self.model_info = json.load(f)
            
            self.feature_names = self.model_info['feature_names']
            
            print("Model loaded successfully!")
            print(f"Model type: {self.model_info['model_type']}")
            print(f"Features: {self.model_info['n_features']}")
            print(f"Training date: {self.model_info['training_date']}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Please run the data science pipeline first!")
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/')
        def home():
            """Home page with prediction form"""
            html_template = """
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>House Price Predictor - ML Model</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body { 
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        padding: 20px;
                    }
                    .container { 
                        max-width: 1000px; 
                        margin: 0 auto; 
                        background: white;
                        border-radius: 20px;
                        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                        overflow: hidden;
                    }
                    .header { 
                        background: linear-gradient(45deg, #2c3e50, #3498db);
                        color: white; 
                        text-align: center; 
                        padding: 40px 20px;
                    }
                    .header h1 { font-size: 2.5em; margin-bottom: 10px; }
                    .header p { font-size: 1.2em; opacity: 0.9; }
                    .form-container { padding: 40px; }
                    .form-grid { 
                        display: grid; 
                        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
                        gap: 25px; 
                        margin-bottom: 30px;
                    }
                    .form-group { margin-bottom: 20px; }
                    label { 
                        display: block; 
                        margin-bottom: 8px; 
                        font-weight: 600; 
                        color: #2c3e50;
                        font-size: 14px;
                    }
                    input, select { 
                        width: 100%; 
                        padding: 12px 15px; 
                        border: 2px solid #e0e6ed; 
                        border-radius: 10px; 
                        font-size: 16px;
                        transition: all 0.3s ease;
                    }
                    input:focus, select:focus { 
                        outline: none; 
                        border-color: #3498db; 
                        box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
                    }
                    .submit-container { text-align: center; margin-top: 30px; }
                    .submit-btn { 
                        background: linear-gradient(45deg, #3498db, #2980b9);
                        color: white; 
                        padding: 15px 40px; 
                        border: none; 
                        border-radius: 50px; 
                        cursor: pointer; 
                        font-size: 18px;
                        font-weight: 600;
                        transition: all 0.3s ease;
                        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.3);
                    }
                    .submit-btn:hover { 
                        transform: translateY(-2px);
                        box-shadow: 0 8px 25px rgba(52, 152, 219, 0.4);
                    }
                    .result { 
                        margin-top: 30px; 
                        padding: 25px; 
                        background: linear-gradient(45deg, #2ecc71, #27ae60);
                        border-radius: 15px; 
                        color: white;
                        text-align: center;
                        box-shadow: 0 5px 15px rgba(46, 204, 113, 0.3);
                    }
                    .result h3 { margin-bottom: 15px; font-size: 1.5em; }
                    .price-display { 
                        font-size: 2.5em; 
                        font-weight: bold; 
                        margin: 15px 0;
                        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
                    }
                    .loading { 
                        display: none; 
                        text-align: center; 
                        margin-top: 20px; 
                    }
                    .spinner { 
                        border: 4px solid #f3f3f3; 
                        border-top: 4px solid #3498db; 
                        border-radius: 50%; 
                        width: 40px; 
                        height: 40px; 
                        animation: spin 1s linear infinite; 
                        margin: 0 auto;
                    }
                    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
                    .model-info {
                        background: #f8f9fa;
                        padding: 20px;
                        border-radius: 10px;
                        margin-bottom: 30px;
                        border-left: 4px solid #3498db;
                    }
                    .model-info h4 { color: #2c3e50; margin-bottom: 10px; }
                    .model-info p { color: #666; margin: 5px 0; }
                    @media (max-width: 768px) { 
                        .form-grid { grid-template-columns: 1fr; }
                        .header h1 { font-size: 2em; }
                        .header p { font-size: 1em; }
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🏠 AI House Price Predictor</h1>
                        <p>Get instant property valuations using advanced machine learning</p>
                    </div>
                    
                    <div class="form-container">
                        <div class="model-info">
                            <h4>📊 Model Information</h4>
                            <p><strong>Algorithm:</strong> {{ model_type }}</p>
                            <p><strong>Features:</strong> {{ n_features }} predictive features</p>
                            <p><strong>Status:</strong> Ready for predictions</p>
                        </div>
                        
                        <form id="predictionForm">
                            <div class="form-grid">
                                <div class="form-group">
                                    <label for="bedrooms">🛏️ Bedrooms:</label>
                                    <select id="bedrooms" name="bedrooms" required>
                                        <option value="1">1 Bedroom</option>
                                        <option value="2">2 Bedrooms</option>
                                        <option value="3" selected>3 Bedrooms</option>
                                        <option value="4">4 Bedrooms</option>
                                        <option value="5">5+ Bedrooms</option>
                                    </select>
                                </div>
                                
                                <div class="form-group">
                                    <label for="bathrooms">🚿 Bathrooms:</label>
                                    <select id="bathrooms" name="bathrooms" required>
                                        <option value="1">1 Bathroom</option>
                                        <option value="2" selected>2 Bathrooms</option>
                                        <option value="3">3 Bathrooms</option>
                                        <option value="4">4+ Bathrooms</option>
                                    </select>
                                </div>
                                
                                <div class="form-group">
                                    <label for="sqft_living">📐 Living Area (sq ft):</label>
                                    <input type="number" id="sqft_living" name="sqft_living" 
                                           value="2000" min="500" max="10000" step="50" required>
                                </div>
                                
                                <div class="form-group">
                                    <label for="sqft_lot">🏞️ Lot Size (sq ft):</label>
                                    <input type="number" id="sqft_lot" name="sqft_lot" 
                                           value="7500" min="1000" max="50000" step="100" required>
                                </div>
                                
                                <div class="form-group">
                                    <label for="floors">🏢 Floors:</label>
                                    <select id="floors" name="floors" required>
                                        <option value="1" selected>1 Floor</option>
                                        <option value="1.5">1.5 Floors</option>
                                        <option value="2">2 Floors</option>
                                        <option value="2.5">2.5 Floors</option>
                                        <option value="3">3+ Floors</option>
                                    </select>
                                </div>
                                
                                <div class="form-group">
                                    <label for="waterfront">🌊 Waterfront Property:</label>
                                    <select id="waterfront" name="waterfront" required>
                                        <option value="0" selected>No Waterfront</option>
                                        <option value="1">Waterfront Property</option>
                                    </select>
                                </div>
                                
                                <div class="form-group">
                                    <label for="view">👀 View Quality (0-4):</label>
                                    <select id="view" name="view" required>
                                        <option value="0" selected>0 - No View</option>
                                        <option value="1">1 - Fair View</option>
                                        <option value="2">2 - Average View</option>
                                        <option value="3">3 - Good View</option>
                                        <option value="4">4 - Excellent View</option>
                                    </select>
                                </div>
                                
                                <div class="form-group">
                                    <label for="condition">🔧 Property Condition:</label>
                                    <select id="condition" name="condition" required>
                                        <option value="1">1 - Poor</option>
                                        <option value="2">2 - Fair</option>
                                        <option value="3" selected>3 - Average</option>
                                        <option value="4">4 - Good</option>
                                        <option value="5">5 - Excellent</option>
                                    </select>
                                </div>
                                
                                <div class="form-group">
                                    <label for="grade">⭐ Construction Grade (3-13):</label>
                                    <input type="number" id="grade" name="grade" 
                                           min="3" max="13" value="7" required>
                                </div>
                                
                                <div class="form-group">
                                    <label for="age">📅 Property Age (years):</label>
                                    <input type="number" id="age" name="age" 
                                           min="0" max="100" value="20" required>
                                </div>
                                
                                <div class="form-group">
                                    <label for="renovated">🔨 Recently Renovated:</label>
                                    <select id="renovated" name="renovated" required>
                                        <option value="0" selected>No Recent Renovation</option>
                                        <option value="1">Recently Renovated</option>
                                    </select>
                                </div>
                                
                                <div class="form-group">
                                    <label for="lat">📍 Latitude:</label>
                                    <input type="number" id="lat" name="lat" 
                                           step="0.0001" value="47.5" required>
                                </div>
                                
                                <div class="form-group">
                                    <label for="long">📍 Longitude:</label>
                                    <input type="number" id="long" name="long" 
                                           step="0.0001" value="-122.3" required>
                                </div>
                            </div>
                            
                            <div class="submit-container">
                                <button type="submit" class="submit-btn">
                                    🚀 Predict House Price
                                </button>
                            </div>
                        </form>
                        
                        <div class="loading" id="loading">
                            <div class="spinner"></div>
                            <p>Analyzing property features...</p>
                        </div>
                        
                        <div id="result" class="result" style="display: none;">
                            <h3>🎯 Price Prediction</h3>
                            <div class="price-display" id="prediction-text"></div>
                            <p>Estimated market value based on current property features</p>
                        </div>
                    </div>
                </div>
                
                <script>
                    document.getElementById('predictionForm').addEventListener('submit', async function(e) {
                        e.preventDefault();
                        
                        // Show loading spinner
                        document.getElementById('loading').style.display = 'block';
                        document.getElementById('result').style.display = 'none';
                        
                        const formData = new FormData(e.target);
                        const data = Object.fromEntries(formData.entries());
                        
                        try {
                            const response = await fetch('/predict', {
                                method: 'POST',
                                headers: { 'Content-Type': 'application/json' },
                                body: JSON.stringify(data)
                            });
                            
                            const result = await response.json();
                            
                            // Hide loading spinner
                            document.getElementById('loading').style.display = 'none';
                            
                            if (result.success) {
                                document.getElementById('prediction-text').innerHTML = 
                                    `${result.prediction.toLocaleString()}`;
                                document.getElementById('result').style.display = 'block';
                                
                                // Smooth scroll to result
                                document.getElementById('result').scrollIntoView({ 
                                    behavior: 'smooth', 
                                    block: 'center' 
                                });
                            } else {
                                alert('❌ Prediction Error: ' + result.error);
                            }
                        } catch (error) {
                            document.getElementById('loading').style.display = 'none';
                            alert('❌ Network Error: ' + error.message);
                        }
                    });
                    
                    // Add input validation and formatting
                    document.querySelectorAll('input[type="number"]').forEach(input => {
                        input.addEventListener('blur', function() {
                            if (this.value) {
                                // Format large numbers with commas for display
                                if (this.name === 'sqft_living' || this.name === 'sqft_lot') {
                                    const value = parseInt(this.value);
                                    if (!isNaN(value)) {
                                        this.value = value;
                                    }
                                }
                            }
                        });
                    });
                </script>
            </body>
            </html>
            """.replace('{{ model_type }}', self.model_info['model_type'] if self.model_info else 'Unknown') \
            .replace('{{ n_features }}', str(self.model_info['n_features']) if self.model_info else 'Unknown')

            return html_template
        
        @self.app.route('/predict', methods=['POST'])
        def predict():
            """API endpoint for making predictions"""
            try:
                data = request.json
                
                # Extract features in the correct order
                features = []
                for feature_name in self.feature_names:
                    if feature_name == 'total_rooms':
                        # Calculate total_rooms
                        total_rooms = int(data['bedrooms']) + int(data['bathrooms'])
                        features.append(total_rooms)
                    elif feature_name == 'is_luxury':
                        # Calculate is_luxury
                        is_luxury = 1 if int(data['grade']) >= 10 else 0
                        features.append(is_luxury)
                    elif feature_name == 'lot_ratio':
                        # Calculate lot_ratio
                        lot_ratio = float(data['sqft_lot']) / float(data['sqft_living'])
                        features.append(lot_ratio)
                    elif feature_name == 'renovated_old':
                        # Calculate renovated_old
                        renovated_old = 1 if (int(data['renovated']) == 1 and int(data['age']) > 30) else 0
                        features.append(renovated_old)
                    else:
                        features.append(float(data[feature_name]))
                
                # Convert to numpy array and reshape
                features_array = np.array(features).reshape(1, -1)
                
                # Scale features if using linear regression
                if hasattr(self.model, 'coef_'):  # Linear regression
                    features_scaled = self.scaler.transform(features_array)
                    prediction = self.model.predict(features_scaled)[0]
                else:  # Tree-based models
                    prediction = self.model.predict(features_array)[0]
                
                # Ensure prediction is positive
                prediction = max(prediction, 50000)
                
                return jsonify({
                    'success': True,
                    'prediction': round(float(prediction), 0),
                    'model_type': self.model_info['model_type'],
                    'confidence': 'High' if hasattr(self.model, 'feature_importances_') else 'Medium'
                })
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({
                    'success': False,
                    'error': str(e)
                })
        
        @self.app.route('/api/model-info')
        def model_info():
            """API endpoint for model information"""
            return jsonify(self.model_info)
        
        @self.app.route('/health')
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'model_loaded': self.model is not None,
                'timestamp': pd.Timestamp.now().isoformat()
            })
    
    def run(self, host='127.0.0.1', port=5000, debug=False):
        """Run the Flask application"""
        print(f"\n🚀 Starting Flask app at http://{host}:{port}")
        print("📊 House Price Predictor is ready!")
        print("🌐 Visit the URL to use the web interface")
        print("📱 Mobile-friendly responsive design")
        print("⚡ Real-time predictions powered by ML")
        self.app.run(host=host, port=port, debug=debug)

def main():
    """Main function to run the complete project"""
    print("="*70)
    print("TASK 3: COMPLETE END-TO-END DATA SCIENCE PROJECT")
    print("HOUSE PRICE PREDICTION WITH DEPLOYMENT")
    print("="*70)
    
    print("\nSelect an option:")
    print("1. 📊 Run complete data science pipeline")
    print("2. 🌐 Start Flask web application")
    print("3. 🚀 Run both (pipeline + deployment)")
    
    choice = input("\nEnter your choice (1/2/3): ").strip()
    
    if choice in ['1', '3']:
        print("\n" + "="*50)
        print("PHASE 1: DATA SCIENCE PIPELINE")
        print("="*50)
        
        # Run data science pipeline
        ds_project = HousePriceDataScience()
        results = ds_project.run_complete_pipeline()
        
        print("\n✅ Data science pipeline completed successfully!")
        print("📁 Generated files:")
        print("   - eda_analysis.png (Exploratory Data Analysis)")
        print("   - model_results.png (Model Performance)")
        print("   - models/ (Trained models and artifacts)")
    
    if choice in ['2', '3']:
        print("\n" + "="*50)
        print("PHASE 2: MODEL DEPLOYMENT")
        print("="*50)
        
        # Start Flask app
        flask_app = HousePriceFlaskApp()
        
        if flask_app.model is None:
            print("❌ Model not found!")
            print("Please run the data science pipeline first (option 1).")
            return
        
        print("\n🎉 Web application is ready!")
        print("="*50)
        print("🏠 House Price Predictor Dashboard")
        print("📊 Real-time ML predictions")
        print("🎨 Beautiful responsive UI")
        print("📱 Mobile-friendly design")
        print("⚡ Instant results")
        
        try:
            flask_app.run(debug=False)
        except KeyboardInterrupt:
            print("\n👋 Shutting down Flask app...")
            print("Thank you for using the House Price Predictor!")

if __name__ == "__main__":
    main()