import pandas as pd
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier 
from sklearn.tree import plot_tree


def clean_bet_data(df) :
    df['Start Time'] = pd.to_datetime(df['Start Time'], utc=True, errors= 'coerce')
    df['Date'] = df['Start Time'].dt.date
    df['Day of Week'] = df['Start Time'].dt.dayofweek
    df['Pick Desc'] = df['Pick Desc'].fillna('custom pick')
    df = df[df['Result'].str.lower() != 'push']
    df = df[df['Odds'].between(-300, 300)]
    df = df.rename(columns={'Odds/Spread/Total': 'Spread/Total'})
    success = []
    for result in df['Result']:
        if result == 'win':
            success.append(1)
        else:
            success.append(0)
    df['Success'] = success
    df = df.drop(columns=['Unnamed: 13'])
    return df
def create_day_mapping():
    day_mapping = pd.DataFrame({'Day of Week': [0, 1, 2, 3, 4, 5, 6], 'Day Name': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']})
    return day_mapping
def merge_day_mapping(df, day_mapping):
    df = df.merge(day_mapping, on='Day of Week', how='left')
    return df
def aggregate(df):
    total_wins = df[df['Success']==1].shape[0]
    total_losses= df[df['Success']==0].shape[0]
    summary= df.groupby('Result')[['Units Wagered', 'Units Net', 'Money Wagered', 'Money Net']].sum().reset_index()
    return total_wins, total_losses, summary
def DOW(df):
    week_types = []
    for day in df['Day Name']:
        if day in ['Saturday', 'Sunday']:
            week_types.append('Weekend')
        else:
            week_types.append('Weekday')
    df['Week Type'] = week_types
    return df
def win_number(df):
    success = []
    for result in df['Result']:
        if result == 'win':
            success.append(1)
        else:
            success.append(0)
    df['Success'] = success
    return df
def day_of_week_win_rate(df):
    weekday = df[df['Week Type'] == 'Weekday']['Success']
    weekend = df[df['Week Type'] == 'Weekend']['Success']
    t_stat, p_value = ttest_ind(weekday, weekend, equal_var=False)
    return {"t_stat": t_stat, "p_value": p_value, "conclusion": ("Reject Null Hypothesis" if p_value < 0.005 else "Fail to reject null hypothesis")}
def countries(df):
    american_leagues = ['nfl', 'nba', 'mlb', 'ncaaf', 'ncaab', 'mls', 'ncaaw']
    regions = []
    for league in df['League']:
        if league in american_leagues:
            regions.append('American')
        else:
            regions.append('Non-American')
    df['Sport Region'] = regions
    return df
def country_winrate(df):
    american = df[df['Sport Region'] == 'American']['Success']
    nonamerican = df[df['Sport Region'] == 'Non-American']['Success']
    t_stat, p_value = ttest_ind(american, nonamerican, equal_var=False)
    return {"t_stat": t_stat, "p_value": p_value, "conclusion": ("Reject Null Hypothesis" if p_value < 0.15 else "Fail to reject null hypothesis")}
def plot_avg_daily_return_by_day(df):
    df['Date'] = pd.to_datetime(df['Date'])
    daily_totals = df.groupby(['Date', 'Day Name'])['Money Net'].sum().reset_index()

    avg_by_day = daily_totals.groupby('Day Name')['Money Net'].mean().reindex([
        'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
    ]).reset_index()
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Day Name', y='Money Net', data=avg_by_day)
    plt.title('Average Daily Return by Day of the Week')
    plt.xlabel('Day of Week')
    plt.ylabel('Average Daily Return ($)')
    plt.tight_layout()
    plt.show()
def plot_league_return_barplot(df, cap=1000):
    df['Region_League'] = df['Sport Region'] + ' - ' + df['League']
    df_plot = df[df['Money Net'].between(-cap, cap)]
    plt.figure(figsize=(15, 8))
    sns.barplot(data=df_plot, x='Region_League', y='Money Net')
    plt.title(f'Distribution of Returns by League (Capped at ±{cap})')
    plt.ylabel('Average Money Net ($)')
    plt.xlabel('Region - League')
    plt.xticks(rotation=20)
    plt.show()
def encode_split(df):
    df = df.drop(columns=['Start Time', 'Date','Pick Desc', 'Result', 'Money Net', 'Units Net', 'Period', 'Type', 'Spread/Total', 'Game', 'Day Name', 'Units Wagered'])
    category_cols = df.select_dtypes(include=['object', 'category']).columns
    le = LabelEncoder()
    for col in category_cols:
        df[col] = le.fit_transform(df[col])
    X = df.drop('Success', axis=1)
    y = df['Success']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return df, X_train_scaled, X_test_scaled, y_train, y_test, X_train, X_test
def correlation_with_success(df):
    le = LabelEncoder()
    df['League'] = le.fit_transform(df['League'])
    df_numeric = df.select_dtypes(include='number')
    corr_with_success = df_numeric.corr()[['Success']].sort_values(by='Success', ascending=False)
    plt.figure(figsize=(6, 6))
    sns.heatmap(corr_with_success, annot=True, fmt=".2f", cmap="coolwarm", cbar=False, linewidths=0.5)
    plt.title("Correlation with Success (including League)")
    plt.tight_layout()
    plt.show()

def kNN_training(X_train_scaled, X_test_scaled, y_train, y_test):
    best_k = None
    best_accuracy = 0
    best_matrix = None
    for k in range(1,21) :
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        y_prediction = knn.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_prediction)
        if accuracy > best_accuracy:
            best_k = k
            best_accuracy = accuracy
            best_matrix = confusion_matrix(y_test, y_prediction)
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_train_scaled, y_train)
    y_final_prediction = best_knn.predict(X_test_scaled)
    final_accuracy = accuracy_score(y_test, y_final_prediction)
    final_matrix = confusion_matrix(y_test, y_final_prediction)
    return best_k, final_accuracy, final_matrix
def decision_tree(X_train, X_test, y_train, y_test):
    d_tree = DecisionTreeClassifier(max_depth=4,random_state=0)
    d_tree.fit(X_train, y_train, sample_weight=None, check_input=True)
    y_predict = d_tree.predict(X_test, check_input=True)
    accuracy = accuracy_score(y_test, y_predict)
    matrix = confusion_matrix(y_test, y_predict)
    return accuracy, matrix, d_tree
def plot_decision_tree(model, feature_names, max_depth=4):
    plt.figure(figsize=(20, 10))
    plot_tree(model,feature_names=feature_names, class_names=['Loss', 'Win'],filled=True,rounded=True, max_depth=max_depth)
    plt.title("Decision Tree Visualization")
    plt.tight_layout()
    plt.show()

