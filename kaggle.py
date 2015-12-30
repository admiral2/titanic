import pandas as pd

from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV
import scorereport

from sklearn.svm import SVC
import numpy as np

import learning_curve
import roc_auc
from sklearn.model_selection import cross_val_score

import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

import re
# A function to get the title from a name.
def get_title(name):
    # Use a regular expression to search for a title.  Titles always consist of capital and lowercase letters, and end with a period.
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
        return title_search.group(1)
    return ""

import operator

# A dictionary mapping family name to id
family_id_mapping = {}

# A function to get the id given a row
def get_family_id(row):
    # Find the last name by splitting on a comma
    last_name = row["Name"].split(",")[0]
    # Create the family id
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    # Look up the id in the mapping
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            # Get the maximum id from the mapping and add one to it if we don't have an id
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]

def fear_per_person(row):
    if row['Fare_org']:
        return row['Fare_org']/(row['SibSp'] + row['Parch'] + 1)

    return None

#replacing all titles with mr, mrs, miss, master
def replace_titles(x):
    title = x['Title']
    if title in ['Mr','Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
        return 'Mr'
    elif title in ['Master']:
        return 'Master'
    elif title in ['Countess', 'Mme','Mrs', 'Dona']:
        return 'Mrs'
    elif title in ['Mlle', 'Ms','Miss']:
        return 'Miss'
    elif title =='Dr':
        if x['Sex']==0:
            return 'Mr'
        else:
            return 'Mrs'
    elif title =='':
        if x['Sex']==0:
            return 'Master'
        else:
            return 'Miss'
    else:
        return title

def cabin_number(x):
    match = re.compile("([0-9]+)").search(x)
    if match:
        return match.group(0)

    return 0

def read_data(filename):
    return pd.read_csv(filename)


def calc_statistic(data, options):
    statistic = {
        "fare": {1: 0, 2: 0, 3: 0},
        "fare_per_person": {1: 0, 2: 0, 3: 0}
    }

    statistic['fare'][1] = np.median(data[data['Pclass'] == 1]['Fare'].dropna())
    statistic['fare'][2] = np.median(data[data['Pclass'] == 2]['Fare'].dropna())
    statistic['fare'][3] = np.median(data[data['Pclass'] == 3]['Fare'].dropna())

    #Fare per person
    data["Fare_org"] = data["Fare"]
    data['Fare_Per_Person'] = data.apply(fear_per_person, axis=1)

    statistic['fare_per_person'][1] = np.median(data[data['Pclass'] == 1]['Fare_Per_Person'].dropna())
    statistic['fare_per_person'][2] = np.median(data[data['Pclass'] == 2]['Fare_Per_Person'].dropna())
    statistic['fare_per_person'][3] = np.median(data[data['Pclass'] == 3]['Fare_Per_Person'].dropna())

    return statistic

def string_to_numbers(data):
    # sex
    data.loc[data["Sex"] == "male", "Sex"] = 0
    data.loc[data["Sex"] == "female", "Sex"] = 1

    # embarked
    data.loc[data["Embarked"] == "S", "Embarked"] = 0
    data.loc[data["Embarked"] == "C", "Embarked"] = 1
    data.loc[data["Embarked"] == "Q", "Embarked"] = 2



    return data

def fill_na(data, statistic):
    # embarked
    data["Embarked"] = data["Embarked"].fillna("S")

    # fill fear ~ median to the pclass
    data.loc[ (data.Fare.isnull())&(data.Pclass==1),'Fare'] = statistic['fare'][1]
    data.loc[ (data.Fare.isnull())&(data.Pclass==2),'Fare'] = statistic['fare'][2]
    data.loc[ (data.Fare.isnull())&(data.Pclass==3),'Fare'] = statistic['fare'][3]

    # Replace missing values with "U0"
    data["Cabin"] = data["Cabin"].fillna("U0")

    data['AgeFill'] = data['Age']
    mean_ages = np.zeros(4)
    mean_ages[0]=np.average(train_data[train_data['Title2'] == 'Miss']['Age'].dropna())
    mean_ages[1]=np.average(train_data[train_data['Title2'] == 'Mrs']['Age'].dropna())
    mean_ages[2]=np.average(train_data[train_data['Title2'] == 'Mr']['Age'].dropna())
    mean_ages[3]=np.average(train_data[train_data['Title2'] == 'Master']['Age'].dropna())
    data.loc[ (data.Age.isnull()) & (data.Title2 == 'Miss') ,'AgeFill'] = mean_ages[0]
    data.loc[ (data.Age.isnull()) & (data.Title2 == 'Mrs') ,'AgeFill'] = mean_ages[1]
    data.loc[ (data.Age.isnull()) & (data.Title2 == 'Mr') ,'AgeFill'] = mean_ages[2]
    data.loc[ (data.Age.isnull()) & (data.Title2 == 'Master') ,'AgeFill'] = mean_ages[3]

    return data

def create_new_features(data):
     # Get all the titles and print how often each one occurs.
    titles = data["Name"].apply(get_title)
    data["Title"] = titles
    data['Title2'] = data.apply(replace_titles, axis=1)
    #print(pd.value_counts(titles))

    # Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
    title_mapping = {"Mr": 1, "Miss": 2, "Ms": 2, "Mrs": 3, "Master": 4, \
                     "Dr": 5, "Rev": 6, "Major": 7, "Capt": 7, "Col": 7, \
                     "Mlle": 8, "Mme": 8, "Don": 9, "Sir": 9, "Lady": 10, \
                     "Countess": 10, "Jonkheer": 10, "Dona": 11}
    for k,v in title_mapping.items():
        titles[titles == k] = v

    # Verify that we converted everything.
    #print(pd.value_counts(titles))

    # Add in the title column.
    data["Title"] = titles

    #Creating new family_size column
    data['FamilySize'] = data['SibSp'] + data['Parch']
    data['Family'] = data['SibSp'] * data['Parch']
    data['FamilySizePClass'] = data['FamilySize'] * data['Pclass']

    # Get the family ids with the apply method
    family_ids = data.apply(get_family_id, axis=1)

    # There are a lot of family ids, so we'll compress all of the families under 3 members into one code.
    family_ids[data["FamilySize"] < 3] = -1

    # Print the count of each unique id.
    #print(pd.value_counts(family_ids))

    data["FamilyId"] = family_ids

    #Fare per person
    data["Fare_org"] = data["Fare"]
    data['Fare_Per_Person'] = data.apply(fear_per_person, axis=1)

    data.loc[ (data.Fare_Per_Person.isnull())&(data.Pclass==1),'Fare_Per_Person'] = statistic['fare_per_person'][1]
    data.loc[ (data.Fare_Per_Person.isnull())&(data.Pclass==2),'Fare_Per_Person'] = statistic['fare_per_person'][2]
    data.loc[ (data.Fare_Per_Person.isnull())&(data.Pclass==3),'Fare_Per_Person'] = statistic['fare_per_person'][3]

    data['AgeCat'] = data['AgeFill']
    data.loc[ (data.Age<=10) ,'AgeCat'] = 1
    data.loc[ (data.Age>10) & (data.Age <= 16) ,'AgeCat'] = 2
    data.loc[ (data.Age>16) & (data.Age <= 30) ,'AgeCat'] = 3
    data.loc[ (data.Age>30) & (data.Age <= 60) ,'AgeCat'] = 4
    data.loc[ (data.Age>60), 'AgeCat'] = 5

    data['AgeClass'] = data['AgeFill'] * data['Pclass']
    data['AgeCatClass'] = data['AgeCat'] * data['Pclass']
    data['PclassFare_Per_Person'] = data['Fare_Per_Person'] * data['Pclass']

    data['SurviveFirst'] = 0
    data.loc[ (data.Sex==1) | (data.Age <=16) ,'SurviveFirst'] = 1
    data.loc[ (data.Sex==0) | (data.Age >50) ,'SurviveFirst'] = 2

    data['SurviveFirst_PClass'] = data['SurviveFirst'] * data['Pclass']

    # convert the distinct cabin letters with incremental integer values
    data['CabinId'] = pd.factorize(data['Cabin'])[0]

    # create feature for the alphabetical part of the cabin number
    data['CabinLetter'] = data['Cabin'].map( lambda x : re.compile("([a-zA-Z]+)").search(x).group())

    # convert the distinct cabin letters with incremental integer values
    data['CabinLetter'] = pd.factorize(data['CabinLetter'])[0]

    data['CabinNumber'] = data['Cabin'].apply(cabin_number)

    return data

def drop_useless_columns(data):
    data = data.drop(['Cabin','Ticket', 'Name', 'Fare_org', 'Age' ,'Title2'], axis=1)

    return data

def drop_high_correlated_features(data, skip_columns):
    df_corr = data.drop(skip_columns,axis=1).corr(method='spearman')

    # create a mask to ignore self-
    mask = np.ones(df_corr.columns.size) - np.eye(df_corr.columns.size)
    df_corr = mask * df_corr

    drops = []
    # loop through each variable
    for col in df_corr.columns.values:
        # if we've already determined to drop the current variable, continue
        if np.in1d([col],drops):
            continue

        # find all the variables that are highly correlated with the current variable
        # and add them to the drop list
        corr = df_corr[abs(df_corr[col]) > 0.98].index
        drops = np.union1d(drops, corr)

    print("\nDropping {} highly correlated features...\n{}".format(drops.shape[0], drops))
    data.drop(drops, axis=1, inplace=True)

    return data

def feature_importance(train_data, alg, fi_threshold = 15, plot = False):
    features_list = train_data.columns.values[2::]
    print(features_list)

    X = train_data[features_list]
    y = train_data["Survived"]

    # Fit a random forest with (mostly) default parameters to determine feature importance
    alg.fit(X, y)
    feature_importance = alg.feature_importances_

    # make importances relative to max importance
    feature_importance = 100.0 * (feature_importance / feature_importance.max())

    # A threshold below which to drop features from the final data set. Specifically, this number represents
    # the percentage of the most important feature's importance value
    #fi_threshold = 15

    # Get the indexes of all features over the importance threshold
    important_idx = np.where(feature_importance > fi_threshold)[0]

    # Create a list of all the feature names above the importance threshold
    important_features = features_list[important_idx]
    print("\n{} Important features(>{}% of max importance):\n{}".format(important_features.shape[0], fi_threshold, \
            important_features))

    # Get the sorted indexes of important features
    sorted_idx = np.argsort(feature_importance[important_idx])[::-1]
    print("\nFeatures sorted by importance (DESC):\n", important_features[sorted_idx])

    # Adapted from http://scikit-learn.org/stable/auto_examples/ensemble/plot_gradient_boosting_regression.html
    if plot:
        import matplotlib.pyplot as plt
        pos = np.arange(sorted_idx.shape[0]) + .5
        plt.subplot(1, 2, 2)
        plt.barh(pos, feature_importance[important_idx][sorted_idx[::-1]], align='center')
        plt.yticks(pos, important_features[sorted_idx[::-1]])
        plt.xlabel('Relative Importance')
        plt.title('Variable Importance')
        plt.draw()
        plt.show()

    return important_features

def gen_learning_curve(alg, X, y, title, n_iter=10, test_size=0.25):
    print("\nCalculating Learning Curve...")
    cv = ShuffleSplit(n_iter=n_iter, test_size=test_size, \
                      random_state=np.random.randint(0,123456789))

    midpoint, diff = learning_curve.plot_learning_curve(alg, title, X, y, (0.6, 1.01), cv=cv, n_jobs=-1)
    return midpoint, diff

def gen_roc_curve(alg, X, y, n_classes=5):
    print("\nGenerating ROC curve 5 times to get mean AUC with class weights...")
    roc_auc.generate_roc_curve(alg, X, y, plot=True, n_classes=n_classes)

# (adapted from http://scikit-learn.org/stable/auto_examples/randomized_search.html)
def report(grid_scores, n_top=5):
    params = None
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Parameters with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.4f} (std: {1:.4f})".format(
              score.mean_validation_score, np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")

        if params == None:
            params = score.parameters

    return params

if __name__ == '__main__':
    train_data = read_data('train.csv')
    test_data = read_data('test.csv')

    all_data = pd.concat([train_data, test_data])

    statistic = calc_statistic(all_data, None)
    train_data = fill_na(train_data, statistic)
    train_data = string_to_numbers(train_data)
    train_data = create_new_features(train_data)
    train_data = drop_useless_columns(train_data)

    train_data.to_csv("train_data_post.csv", index=False)

    alg = RandomForestClassifier(oob_score=True, random_state=np.random.randint(0,123456789), \
                criterion='gini', max_depth=5, n_estimators=150, min_samples_split=4, min_samples_leaf=2, \
                bootstrap=True, max_features='auto')

    #predictors = feature_importance(train_data, alg, plot=False, fi_threshold=15)

    predictors = ['Sex', 'PclassFare_Per_Person',  'AgeFill', 'FamilySize', 'SurviveFirst', 'FamilySizePClass']
    X = train_data[predictors]
    y = train_data["Survived"]

    '''
    grid_test1 = { "n_estimators"      : [150, 1000, 2500],
           "criterion"         : ["gini", "entropy"],
           "max_depth"         : [5, 10, 25],
           "min_samples_split" : [2, 5, 10] }
    print("Hyperparameter optimization using GridSearchCV...")
    grid_search = GridSearchCV(alg, grid_test1, n_jobs=-1, cv=10)
    grid_search.fit(X, y)
    best_params_from_grid_search = scorereport.report(grid_search.grid_scores_)

    '''
    gen_learning_curve(alg, X, y, 'VotingClassifier')
    #gen_roc_curve(alg, X, y)

    scores = cross_val_score(alg, X, y, cv=10)
    print(scores.mean())
    print(scores)

    test_data = fill_na(test_data, statistic)
    test_data = string_to_numbers(test_data)
    test_data = create_new_features(test_data)
    test_data = drop_useless_columns(test_data)

    # Train the algorithm using all the training data
    alg.fit(X, y)

    # Make predictions using the test set.
    predictions = alg.predict(test_data[predictors])

    # Create a new dataframe with only the columns Kaggle wants from the dataset.
    submission = pd.DataFrame({
        "PassengerId": test_data["PassengerId"],
        "Survived": predictions
    })
    #print(submission.describe())
    submission.to_csv("kaggle_1.csv", index=False)
