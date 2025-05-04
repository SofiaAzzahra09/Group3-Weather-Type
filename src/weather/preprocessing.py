import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew

def plot_target_distribution(df, target_column):
    if target_column in df.columns:
        print(f"Distribusi Label: {target_column}")
        print(df[target_column].value_counts())
        sns.countplot(data=df, x=target_column, order=df[target_column].value_counts().index)
        plt.xticks(rotation=45)
        plt.title(f"Distribusi Label '{target_column}'")
        plt.tight_layout()
        plt.show()
    else:
        print(f"Kolom '{target_column}' tidak ditemukan.")

def plot_weather_type_boxplots(df, kategorical_columns, numerical_columns, cols_per_row=3):
    for cat_col in kategorical_columns:
        total_plots = len(numerical_columns)
        num_rows = (total_plots // cols_per_row) + int(total_plots % cols_per_row > 0)
        fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(cols_per_row * 5, num_rows * 5))
        axes = axes.flatten()

        for i, num_col in enumerate(numerical_columns):
            sns.boxplot(x=cat_col, y=num_col, data=df, ax=axes[i])
            axes[i].set_title(f"{num_col} by {cat_col}")
            axes[i].tick_params(axis='x', rotation=45)

        for j in range(i+1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()


def boxplot_outliers_per_column(df, columns, cols_per_row=3):
    total_plots = len(columns)
    num_rows = (total_plots // cols_per_row) + int(total_plots % cols_per_row > 0)
    fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(cols_per_row * 5, num_rows * 5))
    axes = axes.flatten()

    for i, col in enumerate(columns):
        sns.boxplot(x=df[col], ax=axes[i])
        axes[i].set_title(f"Boxplot of {col}")
        axes[i].set_xlabel(col)

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def plot_skewness(df, numerical_columns, cols_per_row=3):
    total_plots = len(numerical_columns)
    num_rows = (total_plots // cols_per_row) + int(total_plots % cols_per_row > 0)

    fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(cols_per_row * 5, num_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(numerical_columns):
        sns.histplot(df[col].dropna(), kde=True, ax=axes[i])
        skewness = skew(df[col].dropna())
        axes[i].set_title(f"{col}\nSkewness: {skewness:.2f}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    # Hapus sumbu kosong
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


# def plot_weather_type_bar(df, kategorical_column, numerical_columns):
#     weather_types = df[kategorical_column].unique()

#     for weather in weather_types:
#         weather_data = df[df[kategorical_column] == weather]
#         means = weather_data[numerical_columns].mean()
        
#         plt.figure(figsize=(8, 4))
#         means.plot(kind='bar')
#         plt.title(f"Average Values for {weather} in {kategorical_column}")
#         plt.ylabel('Average')
#         plt.xlabel('Numerical Features')
#         plt.xticks(rotation=45)
#         plt.tight_layout()
#         plt.show()

    # n_cols = len(df[kategorical_column].unique()) 
    # n_rows = len(numerical_columns)
    
    # fig, axes = plt.subplots(nrows=n_rows, ncols=1, figsize=(10, 5*n_rows))
    # plt.tight_layout(pad=5.0) 
    # for i, num_col in enumerate(numerical_columns):
    #     ax = axes[i] 
    #     for weather in df[kategorical_column].unique():
    #         sns.histplot(
    #             data=df[df[kategorical_column] == weather], 
    #             x=num_col, 
    #             kde=True, 
    #             label=weather, 
    #             bins=20, 
    #             ax=ax 
    #         )
        
    #     ax.set_title(f"Distribution of {num_col} by {kategorikal_column}")
    #     ax.set_ylabel('Frequency')
    #     ax.set_xlabel(num_col)
    #     ax.legend(title=kategorikal_column)

    # plt.show()

def plot_weather_type_bar(df, kategorical_column, numerical_columns, cols_per_row=3):
    weather_types = df[kategorical_column].unique()

    for weather in weather_types:
        weather_data = df[df[kategorical_column] == weather]
        means = weather_data[numerical_columns].mean()

        total_plots = len(means)
        num_rows = (total_plots // cols_per_row) + int(total_plots % cols_per_row > 0)

        fig, ax = plt.subplots(1, 1, figsize=(cols_per_row * 4, 5))
        means.plot(kind='bar', ax=ax)
        ax.set_title(f"Average for {weather}")
        ax.set_ylabel("Average")
        ax.set_xlabel("Numerical Features")
        ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()


def clean_outliers_iqr(data, column, factor=1.5):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]


def result_clean_iqr(df, columns, factor=1.5, cols_per_row=3):
    for column in columns:
        df = clean_outliers_iqr(df, column, factor)

    df = df.reset_index(drop=True)

    total_plots = len(columns)
    num_rows = (total_plots // cols_per_row) + int(total_plots % cols_per_row > 0)

    fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(cols_per_row * 5, num_rows * 5))
    axes = axes.flatten()

    for i, column in enumerate(columns):
        sns.boxplot(x=df[column], ax=axes[i])
        axes[i].set_title(f'Box Plot of {column} after Outlier Handling')
        axes[i].set_xlabel(column)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    return df


def plot_correlation_heatmap(df):
    plt.figure(figsize=(10, 6))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Heatmap Korelasi Fitur Numerik")
    plt.tight_layout()
    plt.show()