import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import logging
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataPipeline:
    """
    A comprehensive ETL pipeline for data preprocessing, transformation, and loading
    """
    
    def __init__(self):
        self.scaler = None
        self.label_encoders = {}
        self.preprocessor = None
        self.pipeline = None
        
    def extract_data(self, file_path=None, data=None):
        """
        Extract data from various sources
        """
        logging.info("Starting data extraction...")
        
        if data is not None:
            df = data.copy()
        elif file_path:
            # Auto-detect file type and read
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                df = pd.read_excel(file_path)
            elif file_path.endswith('.json'):
                df = pd.read_json(file_path)
            else:
                raise ValueError("Unsupported file format")
        else:
            # Generate sample data for demonstration
            np.random.seed(42)
            df = pd.DataFrame({
                'age': np.random.randint(18, 80, 1000),
                'income': np.random.normal(50000, 15000, 1000),
                'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),
                'experience': np.random.randint(0, 40, 1000),
                'target': np.random.choice([0, 1], 1000)
            })
            # Introduce some missing values
            df.loc[np.random.choice(df.index, 50), 'income'] = np.nan
            df.loc[np.random.choice(df.index, 30), 'education'] = np.nan
            
        logging.info(f"Data extracted successfully. Shape: {df.shape}")
        return df
    
    def explore_data(self, df):
        """
        Basic data exploration and quality assessment
        """
        logging.info("Performing data exploration...")
        
        print("Dataset Info:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Data types:\n{df.dtypes}")
        print(f"\nMissing values:\n{df.isnull().sum()}")
        print(f"\nBasic statistics:\n{df.describe()}")
        
        return df
    
    def clean_data(self, df):
        """
        Data cleaning operations
        """
        logging.info("Starting data cleaning...")
        
        # Remove duplicates
        initial_shape = df.shape[0]
        df = df.drop_duplicates()
        logging.info(f"Removed {initial_shape - df.shape[0]} duplicate rows")
        
        # Remove outliers using IQR method for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col != 'target':  # Don't remove outliers from target variable
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                initial_count = len(df)
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                logging.info(f"Removed {initial_count - len(df)} outliers from {col}")
        
        return df
    
    def transform_data(self, df, target_column=None):
        """
        Data transformation including encoding, scaling, and feature engineering
        """
        logging.info("Starting data transformation...")
        
        # Separate features and target
        if target_column and target_column in df.columns:
            X = df.drop(columns=[target_column])
            y = df[target_column]
        else:
            X = df
            y = None
        
        # Identify column types
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        # Create preprocessing pipelines
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False))
        ])
        
        # Create column transformer
        if categorical_cols and numerical_cols:
            self.preprocessor = ColumnTransformer([
                ('num', numerical_pipeline, numerical_cols),
                ('cat', categorical_pipeline, categorical_cols)
            ])
        elif numerical_cols:
            self.preprocessor = ColumnTransformer([
                ('num', numerical_pipeline, numerical_cols)
            ])
        elif categorical_cols:
            self.preprocessor = ColumnTransformer([
                ('cat', categorical_pipeline, categorical_cols)
            ])
        
        # Fit and transform the data
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Get feature names after transformation
        if hasattr(self.preprocessor, 'get_feature_names_out'):
            feature_names = self.preprocessor.get_feature_names_out()
        else:
            # Fallback for older sklearn versions
            feature_names = numerical_cols + [f"{col}_encoded" for col in categorical_cols]
        
        # Convert back to DataFrame for better handling
        X_transformed = pd.DataFrame(X_transformed, columns=feature_names)
        
        logging.info(f"Data transformation completed. New shape: {X_transformed.shape}")
        
        return X_transformed, y
    
    def split_data(self, X, y=None, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        """
        logging.info("Splitting data...")
        
        if y is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            logging.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
            return X_train, X_test, y_train, y_test
        else:
            X_train, X_test = train_test_split(
                X, test_size=test_size, random_state=random_state
            )
            logging.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
            return X_train, X_test, None, None
    
    def load_data(self, data, file_path, file_format='csv'):
        """
        Load processed data to various destinations
        """
        logging.info(f"Loading data to {file_path}")
        
        if isinstance(data, tuple):
            # Handle multiple datasets (train/test split)
            X_train, X_test, y_train, y_test = data
            
            if file_format == 'csv':
                X_train.to_csv(f"{file_path}_X_train.csv", index=False)
                X_test.to_csv(f"{file_path}_X_test.csv", index=False)
                if y_train is not None:
                    pd.DataFrame(y_train).to_csv(f"{file_path}_y_train.csv", index=False)
                    pd.DataFrame(y_test).to_csv(f"{file_path}_y_test.csv", index=False)
            elif file_format == 'excel':
                with pd.ExcelWriter(f"{file_path}.xlsx") as writer:
                    X_train.to_excel(writer, sheet_name='X_train', index=False)
                    X_test.to_excel(writer, sheet_name='X_test', index=False)
                    if y_train is not None:
                        pd.DataFrame(y_train).to_excel(writer, sheet_name='y_train', index=False)
                        pd.DataFrame(y_test).to_excel(writer, sheet_name='y_test', index=False)
        else:
            # Handle single dataset
            if file_format == 'csv':
                data.to_csv(f"{file_path}.csv", index=False)
            elif file_format == 'excel':
                data.to_excel(f"{file_path}.xlsx", index=False)
        
        logging.info("Data loading completed")
    
    def run_pipeline(self, input_path=None, output_path="processed_data", 
                    target_column='target', data=None):
        """
        Run the complete ETL pipeline
        """
        logging.info("Starting ETL pipeline...")
        
        try:
            # Extract
            df = self.extract_data(input_path, data)
            
            # Explore
            self.explore_data(df)
            
            # Clean
            df_clean = self.clean_data(df)
            
            # Transform
            X_transformed, y = self.transform_data(df_clean, target_column)
            
            # Split
            if y is not None:
                X_train, X_test, y_train, y_test = self.split_data(X_transformed, y)
                split_data = (X_train, X_test, y_train, y_test)
            else:
                X_train, X_test, _, _ = self.split_data(X_transformed)
                split_data = (X_train, X_test, None, None)
            
            # Load
            self.load_data(split_data, output_path)
            
            logging.info("ETL pipeline completed successfully!")
            return split_data
            
        except Exception as e:
            logging.error(f"Pipeline failed: {str(e)}")
            raise

# Example usage and demonstration
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = DataPipeline()
    
    # Run pipeline with sample data (no input file needed)
    print("Running ETL Pipeline with sample data...")
    print("="*50)
    
    result = pipeline.run_pipeline(
        input_path=None,  # Use sample data
        output_path="processed_sample_data",
        target_column='target'
    )
    
    X_train, X_test, y_train, y_test = result
    
    print(f"\nPipeline Results:")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Target variable shape: {y_train.shape if y_train is not None else 'None'}")
    
    # Example with custom data file (uncomment to use)
    # result = pipeline.run_pipeline(
    #     input_path="your_data.csv",
    #     output_path="your_processed_data",
    #     target_column='your_target_column'
    # )