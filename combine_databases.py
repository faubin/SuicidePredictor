import os
import copy
import pandas as pd
import pylab as pl
import seaborn
# model 1
from sklearn import linear_model
# model 2
import statsmodels.api as sm


def convert_st_to_state(st):
    """
    Returns the state name (string) from the state abreviation (string)
    """
    us_state_abbrev = {'AL': 'Alabama',
                       'AK': 'Alaska',
                       'AZ': 'Arizona',
                       'AR': 'Arkansas',
                       'CA': 'California',
                       'CO': 'Colorado',
                       'CT': 'Connecticut',
                       'DE': 'Delaware',
                       'FL': 'Florida',
                       'GA': 'Georgia',
                       'HI': 'Hawaii',
                       'ID': 'Idaho',
                       'IL': 'Illinois',
                       'IN': 'Indiana',
                       'IA': 'Iowa',
                       'KS': 'Kansas',
                       'KY': 'Kentucky',
                       'LA': 'Louisiana',
                       'ME': 'Maine',
                       'MD': 'Maryland',
                       'MA': 'Massachusetts',
                       'MI': 'Michigan',
                       'MN': 'Minnesota',
                       'MS': 'Mississippi',
                       'MO': 'Missouri',
                       'MT': 'Montana',
                       'NE': 'Nebraska',
                       'NV': 'Nevada',
                       'NH': 'New Hampshire',
                       'NJ': 'New Jersey',
                       'NM': 'New Mexico',
                       'NY': 'New York',
                       'NC': 'North Carolina',
                       'ND': 'North Dakota',
                       'OH': 'Ohio',
                       'OK': 'Oklahoma',
                       'OR': 'Oregon',
                       'PA': 'Pennsylvania',
                       'RI': 'Rhode Island',
                       'SC': 'South Carolina',
                       'SD': 'South Dakota',
                       'TN': 'Tennessee',
                       'TX': 'Texas',
                       'UT': 'Utah',
                       'VT': 'Vermont',
                       'VA': 'Virginia',
                       'WA': 'Washington',
                       'WV': 'West Virginia',
                       'WI': 'Wisconsin',
                       'WY': 'Wyoming'}
    if st not in us_state_abbrev.keys():
        error_text = 'No state matches {0:s}'.format(st)
        raise ValueError(error_text)
    return us_state_abbrev[st]


def select_health_files_to_load(years_to_analyze, path='downloads'):
    """
    Returns a list of files from path that have one of the years (int) in the
    supplied list
    """
    all_files = os.listdir(path)
    files = []
    for file_ in all_files:
        for year in years_to_analyze:
            if '{0:d}.xls'.format(year) in file_:
                files.append(file_)
    return sorted(files)


def load_data(files, state_or_county, path='downloads/'):
    """
    """
    for n_file, file_ in enumerate(files):
        # loading a database
        print('Loading {0:s}'.format(file_))
        xls = os.path.join(path, file_)
        new_df = pd.read_excel(xls, 'Ranked Measure Data', header=1)

        # droping empty lines
        #new_df = new_df.dropna(how='all')  # leaves the last empty line...
        new_df = new_df.drop(pl.where(pd.isnull(new_df["FIPS"].values))[0])

        # adding the year to the data
        year = int(xls.split('_')[1].split('.')[0])
        new_df['Year'] = pd.Series(year*pl.ones(len(new_df.index)),
                                   index=new_df.index)

        # appends the data to the main database
        if n_file == 0:
            df = copy.deepcopy(new_df)
        else:
            df = df.append(new_df)
    # reset index
    df = df.reset_index(drop=True)

    if state_or_county == 'county':
        df = df.drop(pl.where(pd.isnull(df["County"]).values)[0])
    elif state_or_county == 'state':
        df = df.drop(pl.where(~pd.isnull(df["County"]).values)[0])
    else:
        error_text = 'Only "state" or "county supported. You nentered '
        error_text += '{0:s}'.format(state_or_county)
    # reset index
    df = df.reset_index(drop=True)
    return df


def exclude_redundant_data(df):
    """
    """
    keys_to_analyze = ['Year',
                       'FIPS',
                       'State',
                       'County',
                       'Years of Potential Life Lost Rate',
                       '% Fair/Poor',
                       'Physically Unhealthy Days',
                       'Mentally Unhealthy Days',
                       '% LBW',
                       '% Smokers',
                       '% Obese',
                       'Food Environment Index',
                       '% Physically Inactive',
                       '% With Access',
                       '% Excessive Drinking',
                       '% Alcohol-Impaired',
                       'Chlamydia Rate',
                       'Teen Birth Rate',
                       '% Uninsured',
                       'PCP Rate',
                       'Dentist Rate',
                       'MHP Rate',
                       'Preventable Hosp. Rate',
                       '% Receiving HbA1c',
                       '% Mammography',
                       'Graduation Rate',
                       '% Some College',
                       '% Unemployed',
                       '% Children in Poverty',
                       'Income Ratio',
                       '% Single-Parent Households',
                       'Association Rate',
                       'Violent Crime Rate',
                       'Injury Death Rate',
                       'Average Daily PM2.5',
                       'Presence of violation',  # boolean
                       '% Severe Housing Problems',
                       '% Drive Alone',
                       '% Long Commute - Drives Alone',
                       ]
    return df[keys_to_analyze]


def add_suicide_data(df, path='downloads'):
    """
    """
    # adding empty data
    df['Suicide'] = pd.Series(pl.nan*pl.ones(len(df.index)), index=df.index)

    # years to load
    years = set(df['Year'].values.astype(int))
    for year in years:
        file_name = 'SUICIDE{0:d}.csv'.format(year)

        if file_name not in os.listdir(path):
            print("Warning: no suicide data for {0:d}".format(year))
        else:
            # loading data
            suicide_df = pd.read_csv(os.path.join(path, file_name))

            # add the data state by state
            for st in suicide_df["STATE"]:
                state = convert_st_to_state(st)

                index_suicide = pl.where(suicide_df["STATE"].values == st)[0]
                if len(index_suicide) == 1:
                    # index in df
                    valid = df['State'].values == state
                    valid &= df['Year'].values == year
                    index_df = pl.where(valid)[0]
                    if len(index_df) == 1:
                        df.set_value(index_df[0], 'Suicide',
                                     suicide_df['RATE'][index_suicide[0]])
                    else:
                        warn_text = 'Warning: I expect 1 value, got '
                        warn_text += '{0:d} '.format(len(index_df))
                        warn_text += 'value in health database for '
                        warn_text += 'state {0:s}'.format(state)
                        print(warn_text)
                else:
                    warn_text = 'Warning: I expect 1 value, got '
                    warn_text += '{0:d} '.format(len(index_suicide))
                    warn_text += 'value in suicide database for '
                    warn_text += 'state {0:s}'.format(state)
                    print(warn_text)
    return df


def clean_data(df):
    """
    remove columns fron dataframe qhich are undefined, then remove rows with
    missing elements
    """
    # remove empty columns
    df = df.dropna(axis=1, how='all')
    # remove rows with at least 1 element missing
    df = df.dropna(axis=0, how='any')
    # reset index
    df = df.reset_index(drop=True)
    return df


def visualize_data(df, show_plots=True):
    """
    """
    pl.figure(1, figsize=(12, 9))
    pl.axes([0.25, 0.3, 0.95-0.25, 0.95-0.3])
    seaborn.heatmap(df.corr())
    pl.yticks(rotation=0, fontsize=12)
    pl.xticks(rotation=90, fontsize=12)
    pl.title("Correlation between the data", fontsize=24)
    if show_plots:
        pl.show()
    else:
        pl.close(1)
    return


################################################################################
# Parameters
################################################################################
years_to_analyze = [2016]
state_or_county = 'state'
# show_plots = True
show_plots = False

################################################################################
# Loading and handling the data
################################################################################
# selects the Health databases
files = select_health_files_to_load(years_to_analyze)
# load raw database
data = load_data(files, state_or_county)
# select columns
data = exclude_redundant_data(data)
# add a column for suicide rate
data = add_suicide_data(data)
# remove columns and rows with important data missing
data = clean_data(data)

# show the raw data
visualize_data(data, show_plots)

################################################################################
# Modeling
################################################################################
# remove classification data, County not present for state analysis
model_data = copy.deepcopy(data)
states = model_data['State'].values
for key_ in ['Year', 'FIPS', 'State', 'County']:
    if key_ in model_data.keys():
        model_data = model_data.drop([key_], axis=1)

# normalizing
means = {}
stds = {}
for key_ in model_data.keys():
    means[key_] = pl.mean(model_data[key_].values)
    stds[key_] = pl.std(model_data[key_].values)
    model_data[key_] = pd.Series((model_data[key_] - means[key_]) / stds[key_],
                          index=model_data.index)

#model_data = model_data.dropna(axis=1, how='any')

X = model_data.as_matrix()
Y = X[:, -1]
X = X[:, :-1]

#x_train=input_variables_values_training_datasets
#y_train=target_variables_values_training_datasets
#x_test=input_variables_values_test_datasets

# Create linear regression object
#linear = linear_model.LinearRegression()

# Train the model using the training sets and check score
#linear.fit(x_train, y_train)
#linear.score(x_train, y_train)
#linear.fit(X, Y)
#linear.score(X, Y)
#Equation coefficient and Intercept
#print('Coefficient: \n', linear.coef_)
#print('Intercept: \n', linear.intercept_)
#Predict Output
#predicted= linear.predict(X)


model = sm.OLS(Y, X)
results = model.fit()
print results.params
print results.tvalues
predicted = model.predict(results.params, X)

param_names = list(model_data.keys().values)
param_names.remove('Suicide')
print('\n{0:35s}: {1:7s} {2:7s}'.format('parameter name', 'value', 'p-value'))
for i in range(len(param_names)):
    print('{0:35s}: {1:7.4f} {2:7.4f}'.format(param_names[i], results.params[i], results.pvalues[i]))

# denormalizing
Y = Y * stds['Suicide'] + means['Suicide']
predicted = predicted * stds['Suicide'] + means['Suicide']

# plotting results vs data
pl.plot(Y, predicted, '.b')
#if show_plots:
#    pl.show()
#else:
#    pl.close('all')




pl.figure(3, figsize=(8, 12))
pl.subplot(121)
Y_to_plot = [[i] for i in Y]
ax1 = seaborn.heatmap(Y_to_plot, cbar=False)
pl.yticks(rotation=0)
ax1.set_yticklabels(states)
ax1.set_xticklabels([])
pl.xlabel("State")

pl.subplot(122)
predicted_to_plot = [[i] for i in predicted]
ax2 = seaborn.heatmap(predicted_to_plot)
ax2.set_yticklabels([])
ax2.set_xticklabels([])
pl.xlabel("County")

print("Check if my Y and states are inverted...")
pl.show()

