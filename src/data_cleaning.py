import pandas

def standardize_col_names(df):
    '''
    Standardizes column names of a dataframe.
    It will remove white space, replace spaces with underscores, and eliminate special characters (including parenthesis and slashes).
    
    Parameters
    ----------
    
    df : Dataframe object
    
    Return Values
    -------------
    Dataframe with column names standardized.
    '''
    df.columns = (df.columns
                .str.strip()
                .str.lower()
                .str.replace(' ', '_')
                .str.replace('(', '')
                .str.replace(')', '')
                .str.replace('/','')
                .str.replace('\\',''))
    return df
    
        
def null_counts(df):
    '''
    Returns a dataframe containing the number of null values in each column of a given dataframe.
    
    Parameters
    ----------
    df : A DataFrame to check for null values.
    '''
    
    null_df = pandas.DataFrame(df.isnull().sum(), columns=['null_count'])
    null_df['null_fraction'] = null_df['null_count'] / df.shape[0]
    null_df = null_df.sort_values('null_count',ascending=False)
    return null_df
   