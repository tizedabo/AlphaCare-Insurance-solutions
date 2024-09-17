import pandas as pd

def check_missing_data(df):
    """
    Check for missing data in a DataFrame and return a summary of missing values.
    
    Parameters:
    - df: Pandas DataFrame
    
    Returns:
    - DataFrame or str: Summary of missing data or success message
    """
    missing_data = df.isnull().sum()
    missing_data_summary = pd.DataFrame({
        'Column Name': missing_data.index,
        'Missing Values': missing_data.values,
        'Percentage Missing': (missing_data.values / len(df)) * 100
    })
    missing_data_summary = missing_data_summary[missing_data_summary['Missing Values'] > 0]
    
    if missing_data_summary.empty:
        return "Success: No missing values."
    else:
        # Keep only the required columns
        missing_data_summary = missing_data_summary[['Column Name', 'Missing Values', 'Percentage Missing']]
        
        return missing_data_summary

def get_total_missing_percentage(df):
    """
    Calculate the total percentage of missing values from all columns in a DataFrame.
    
    Parameters:
    - df: Pandas DataFrame
    
    Returns:
    - missing_data_percentage: Total percentage of missing values
    """
    # Calculate total missing values across all columns and total percentage of missing values
    total_percentage = (df.isnull().sum().sum() / df.size) * 100
    
    return total_percentage


def check_duplicates(df):
    """
    Check for duplicate rows in a DataFrame and return a summary.
    
    Parameters:
    - df: Pandas DataFrame
    
    Returns:
    - DataFrame or str: Summary of duplicate rows or success message
    """
    # Find duplicate rows
    duplicates = df[df.duplicated(keep='first')]
    
    if duplicates.empty:
        return "Success: No duplicated values."
    else:
        # Get the first column name
        first_column_name = df.columns[0]
        
        # Get the first column value from the duplicated rows
        duplicates_summary = duplicates[[first_column_name]].copy()
        duplicates_summary['Number of Duplicates'] = duplicates.groupby(first_column_name)[first_column_name].transform('count')
        
        # Drop duplicate rows from the summary
        duplicates_summary.drop_duplicates(inplace=True)
        
        return duplicates_summary


def check_data_types(df):
    """
    Check data types of columns in a DataFrame and return a summary.
    
    Parameters:
    - df: Pandas DataFrame
    
    Returns:
    - DataFrame or str: Summary of columns with different data types or success message
    """
    dtypes_summary = df.dtypes.reset_index()
    dtypes_summary.columns = ['Column Name', 'Data Type']
    
    # Group by column name and get the unique data types per column
    grouped = dtypes_summary.groupby('Column Name')['Data Type'].apply(list).reset_index()
    
    # Create a new column for the number of unique data types per column
    grouped['Number of Different Data Types'] = grouped['Data Type'].apply(len)
    
    # Filter out columns with uniform data types
    non_uniform_columns = grouped[grouped['Number of Different Data Types'] > 1].copy()
    
    if non_uniform_columns.empty:
        return "Success: Data types per column are uniform."
    else:
        # Rename the 'Data Type' column to 'List of Data Types'
        non_uniform_columns.rename(columns={'Data Type': 'List of Data Types'}, inplace=True)
        return non_uniform_columns

def get_numeric_columns(df):
    """
    Get a list of column names with numeric data types from a DataFrame.
    
    Parameters:
    - df: Pandas DataFrame
    
    Returns:
    - numeric_columns: List of column names with numeric data types
    """
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    return numeric_columns


def check_numeric_anomalies(df, column, lower_bound=None, upper_bound=None):
    """
    Check for numeric anomalies in a specific column of a DataFrame and return a summary.
    
    Parameters:
    - df: Pandas DataFrame
    - column: The specific column to check
    - lower_bound: Lower bound for numeric anomalies (optional)
    - upper_bound: Upper bound for numeric anomalies (optional)
    
    Returns:
    - str or DataFrame: Success message or summary of anomalies
    """
    if df[column].dtype not in ['int64', 'float64']:
        return f"Error: Column {column} is not numeric."
    
    if lower_bound is not None and upper_bound is not None:
        anomalies = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    else:
        anomalies = df[~df[column].apply(lambda x: isinstance(x, (int, float)))]
    
    if anomalies.empty:
        return "Success: No anomalies detected."
    else:
        anomalies_summary = pd.DataFrame({
            'Column Name': [column],
            'Number of Anomalies': [len(anomalies)]
        })
        return anomalies_summary
