import numpy as np
import pandas as pd
# import library for preprocessing
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# import library for dealing with class imbalance
import imblearn
from imblearn.over_sampling import SMOTE

print("imblearn version " + imblearn.__version__)

# import library for splitting dataset
from sklearn.model_selection import train_test_split

# create list containing categorical columns
cat_cols = ['job', 'marital', 'education', 'default', 'housing',
            'loan', 'contact', 'month', 'day_of_week', 'poutcome']

# create list containing numerical columns
num_cols = ['duration', 'campaign', 'emp.var.rate', "pdays", "age", 'cons.price.idx', 'cons.conf.idx', 'euribor3m',
            'nr.employed', 'previous']


def check_info(data):
    print(data.name, 'dataset schema:')
    print('---------------------------------------------')
    print(data.info())


def shape(data):
    print(data.name, 'shape:', data.shape)


def size(data):
    print(data.name, 'size:', data.size)


def get_unique_values(data):
    cols = data.columns
    for i in cols:
        if data[i].dtype == 'O':
            print('Unique values in column ', i, 'are', data[i].unique())
            print('----------------------------------------------')


def check_missing_val(data):
    print('Sum of missing values in', data.name)
    print('------------------------------')
    print(data.isnull().sum())


def get_categorical_variables(data):
    categorical_data = data.select_dtypes(exclude='number')
    categorical_data.head()


def get_numercial_variables(data):
    categorical_data = data.select_dtypes(exclude='number')
    categorical_data.head()


def replace_outliers_with_nan(num_data, dataset_new):
    # treating outliers
    count = 1
    for col in num_data:
        Q1 = num_data[col].quantile(0.25)
        Q3 = num_data[col].quantile(0.75)
        IQR = Q3 - Q1
        print(f'column {count}: {num_data[col].name}\n------------------------')
        print('1st quantile => ', Q1)
        print('3rd quantile => ', Q3)
        print('IQR =>', IQR)

        fence_low = Q1 - (1.5 * IQR)
        print('fence_low => ' + str(fence_low))

        fence_high = Q3 + (1.5 * IQR)
        print('fence_high => ' + str(fence_high))
        print("\n------------------------")

        count = count + 1

        # replacing outliers with nan
        dataset_new[col][((dataset_new[col] < fence_low) | (dataset_new[col] > fence_high))] = np.nan


def get_column_with_nan_values(dataset_new):
    print(dataset_new.select_dtypes(include='number').isnull().sum())


def export_to_csv(data, name, index=False):
    data.to_csv(name, index=index)


# function to encode categorical columns
def encode(data):
    cat_var_enc = pd.get_dummies(data[cat_cols], drop_first=False)
    return cat_var_enc


# function to
def preprocessed(data):
    # adding the encoded columns to the dataframe
    data = pd.concat([data, encode(data)], axis=1)
    # saving the column names of categorical variables
    cat_cols_all = list(encode(data).columns)
    # creating a new dataframe with features and output
    cols_input = num_cols + cat_cols_all
    preprocessed_data = data[cols_input + ['subscribed']]
    return preprocessed_data


# function to rescale numerical columns
def rescale(data):
    # creating an instance of the scaler object
    scaler = StandardScaler()
    data[num_cols] = scaler.fit_transform(data[num_cols])
    return data


# function to get the dependent and independent variable
def split_input_output_variables(data):
    X = data.drop(columns=["subscribed", 'duration'])
    y = data["subscribed"]
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    return X, y


# function to split dataset
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1)
    # printing the shape of training set
    print(f'Train set X shape: {X_train.shape}')
    print(f'Train set y shape: {y_train.shape}')
    # printing the shape of test set
    print(f'Test set X shape: {X_test.shape}')
    print(f'Test set y shape: {y_test.shape}')
    return X_train, X_test, y_train, y_test


# function to reduce dimensions
def dimension_reduction(method, components, train_data, test_data):
    # PCA
    if (method == 'PCA'):
        pca = PCA(n_components=components)
        pca.fit(train_data)
        pca_train = pca.transform(train_data)
        X_train_reduced = pd.DataFrame(pca_train)
        print("original shape:   ", train_data.shape)
        print("transformed shape:", X_train_reduced.shape)
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
        # applying method transform to X_test
        pca_test = pca.transform(test_data)
        X_test_reduced = pd.DataFrame(pca_test)

    # TSNE
    elif (method == 'TSNE'):
        tsne = TSNE(n_components=components)
        tsne_train = tsne.fit_transform(train_data)
        X_train_reduced = pd.DataFrame(tsne_train)
        print("original shape:   ", train_data.shape)
        print("transformed shape:", X_train_reduced.shape)
        # applying method transform to X_test
        tsne_test = tsne.fit_transform(test_data)
        X_test_reduced = pd.DataFrame(tsne_test)

    else:
        print('Dimensionality reduction method not found!')

    return X_train_reduced, X_test_reduced


# function to deal with imbalanced class
def class_imbalance(X_data, y_data):
    # creating an instance
    sm = SMOTE(random_state=27)
    # applying it to the data
    X_train_smote, y_train_smote = sm.fit_resample(X_data, y_data)
    return X_train_smote, y_train_smote
