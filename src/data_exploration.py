import pandas
import matplotlib
import matplotlib.pyplot as plt
import seaborn

### plot_correlation_matrix_heat_map
#Returns a heatmap showing the N variables that are most correlated with a given column within a dataframe.

#Parameters:
#- df: a Pandas DataFrame to calculate correlation on
#- label: the variable to calculate correlation with
#- qty_fields: the number of variables (N) to display on the heatmap. ie, the dimension of the heatmap.

#This function uses the Python libraries [Pandas](https://pandas.pydata.org/docs/reference/index.html) (pandas) and [Matplotlib](https://matplotlib.org/contents.html) (plt), both of which have been imported above.

def plot_correlation_matrix_heat_map(df,label,qty_fields=10):
    df = pandas.concat([df[label],df.drop(label,axis=1)],axis=1)
    correlation_matrix = df.corr()
    index = correlation_matrix.sort_values(label, ascending=False).index
    correlation_matrix = correlation_matrix[index].sort_values(label,ascending=False)

    fig,ax = plt.subplots()
    fig.set_size_inches((10,10))
    seaborn.heatmap(correlation_matrix.iloc[:qty_fields,:qty_fields],annot=True,fmt='.2f',ax=ax)
    return(fig,ax)


### null_counts
#Returns a dataframe containing the number of null values in each column of a given dataframe.

#Parameters:
#- df: A DataFrame to check for null values.

#This function uses the Python libraries [Pandas](https://pandas.pydata.org/docs/reference/index.html) (pandas), which has been imported above.

def null_counts(df):
    null_df = pandas.DataFrame(df.isnull().sum(),columns=['null_count'])
    null_df['null_fraction'] = null_df['null_count'] / df.shape[0]
    null_df = null_df.sort_values('null_count',ascending=False)
    return null_df