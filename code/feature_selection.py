"""
This Python function, `significant_features`, utilizes the Mann-Whitney U test from the scipy.stats module to identify features 
that exhibit a statistically significant difference between two groups in a given dataset.

Parameters:
- `data`: Pandas DataFrame containing the dataset with a 'class' column indicating group membership (0 for one group, 1 for another).

Functionality:
1. Separates the dataset into two groups based on the 'class' column: 'healthy' (class 0) and 'hf' (class 1).
2. Converts the data of each group into NumPy arrays for analysis.
3. Iterates through each feature column and performs the Mann-Whitney U test to compare the distributions of the two groups.
4. Identifies features with a statistically significant difference (p-value < 0.05) in medians between the two groups.
5. Returns a list of column names corresponding to the significant features.

Note: The Mann-Whitney U test is a non-parametric test used to determine whether there is a difference between two independent distributions. 
The function is particularly useful for identifying features that may play a role in distinguishing between the two groups in the dataset.
"""
import scipy.stats as ss

def significant_features(data):
    healthy = data.loc[data['class'] == 0, data.columns != 'class']
    hf = data.loc[data['class'] == 1, data.columns != 'class']

    healthy_values = healthy.to_numpy()
    hf_values = hf.to_numpy()
    
    significant_difference_columns = []

    for i in range(healthy_values.shape[1]):
        # Perform Mann-Whitney U test
        u_statistic, p_value_mannwhitney = ss.mannwhitneyu(healthy_values[:, i],
                                                           hf_values[:, i], 
                                                           alternative='two-sided')
        
        if p_value_mannwhitney < 0.05:
            # The rejection of null hypothesis for this test posits there is 
            # statistically significant difference in medians of the features 
            # extracted from both groups.
            significant_difference_columns.append(healthy.columns[i])

    return significant_difference_columns