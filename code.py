# Import libraries
import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import classification_report, confusion_matrix

# Landing page
st.header('Marketing and Customer Analytics')
df = pd.read_csv('marketing_data.csv')
          
## DATA CLEANING

# Remove unnecessary columns and rows
df = df.drop(['ID','AcceptedCmp3', 'AcceptedCmp4',
            'AcceptedCmp5', 'AcceptedCmp1',
            'AcceptedCmp2', 'Complain'], axis=1)

# Rename misspelled column
df.rename(columns={' Income ':'Income'}, inplace=True)

# Convert inappropriate dtypes and extract better features
df['Income']= df['Income'].str.replace(',', '').str.strip('$').astype(float)
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
df['Month'] = df['Dt_Customer'].dt.month
df['Year'] = df['Dt_Customer'].dt.year
df = df.drop(['Dt_Customer'], axis=1)

# Remove outliers records
df = df[df.Year_Birth >= 1940]
df = df[df.Income < 107000]

# Reindex columns so that the target variable is last & categoricals are first
df = df.reindex(columns=['Education', 'Marital_Status', 'Country', 'Year_Birth', 'Income', 'Year',
                        'Month','Kidhome','Teenhome', 'Recency', 'MntWines', 'MntFruits',
                        'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                        'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
                        'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth', 'Response'
                        ])

# Rename columns for clarity
df.rename(columns={'Year_Birth':'Birth_Year',
                    'Kidhome':'Kid@Home', 'Teenhome':'Teen@Home',
                    'Year':'Enrol_Year', 'Month':'Enrol_Month'}, inplace=True)

# Impute missing income values with the column mean
df['Income'].fillna((df['Income'].mean()), inplace=True)

# Drop all duplicates in the dataset
dups = df.duplicated()
df.drop_duplicates(inplace=True)

# Rename values in 'Country' and 'Education' columns
df['Country'].replace({'SP':'Spain', 'CA':'Canada', 'US':'United States',
                        'AUS':'Australia', 'GER':'Germany', 'IND':'India',
                        'SA':'South Africa', 'ME':'Montenegro'}, inplace=True)
df['Education'].replace({'Graduation': 'Bachelor', '2n Cycle': 'Maisteri'}, inplace=True)

## DATA OVERVIEW

# Display the number of rows and columns in a sentence
st.write("This dataset has: ", df.shape[0], " records and ", df.shape[1], " features")

# Add a slider that filters total number of records and updates all subsequent values
records = st.sidebar.slider('Select total desired records', df.shape[0], 1, df.shape[0])
df = df.iloc[0:records]

# Display df.head() under an expandable tab
with st.beta_expander("Quick Glance"):
    glance = st.slider('Select desired number of records to display', min_value=1, max_value=100, value=5)
    st.write(df.head(glance))

# Review the summary statistics under an expandable tab
with st.beta_expander("Summary Statistics"):
    st.write(df.describe())

# Create selectbox to choose appropriate section
option = st.sidebar.radio('What would you like to do?',
                ('Nothing', 'Exploratory Analysis', 'Machine Learning'))
                # I want the user to notice and select before anything appears, so the default is 'Nothing'

## EXPLORATORY ANALYSIS

if option == "Exploratory Analysis":               #If this section is selected
    with st.beta_container():                      #Creates a container/box of graphs so they look like a dashboard
        col1, col2= st.beta_columns([1.5,1])       #Creates 2 side-by-side graphs, with col1 slightly bigger than col2

        # Which generation is responding the most to marketing campaigns?
        with col1:
            year = pd.Series(df['Birth_Year'])
            df['Decade_Birth'] = (np.floor(year / 10) * 10).astype('int64')
            responded = df.groupby('Decade_Birth', as_index=False)['Response'].mean()

            fig1 = px.line(responded, x="Decade_Birth", y="Response",
                      title = "Campaign Response Rate by Generation")
            fig1.layout.update(height=450, width=450, xaxis_title='Generation', yaxis_title='Response Rate',
                                xaxis_showgrid=False, yaxis_showgrid=False)
            fig1.update_traces(line=dict(width=2))        #Changes line width
            st.write(fig1)

        # Which countries do most of our customers come from?
        with col2:
            labels = df['Country'].unique()
            values = df['Country'].value_counts()
            fig2 = go.Figure(data=[go.Pie(labels=labels, values=values,
                                     textinfo='label+percent'
                                     )])
            fig2.layout.update(height=450, width=450,showlegend=False,          #Scales graph & removes legend
                            title="Top Countries of Origin", title_x = 0.5,    #Sets title & centers it
                            xaxis_showgrid=False, yaxis_showgrid=False)
            st.write(fig2)


        # Now for the second half of the dashboard
        col3, col4= st.beta_columns([1.5,1])

        # What are the educational levels of our targets?
        with col3:
            fig3 = px.histogram(df, x='Education', color="Education")
            fig3.layout.update(yaxis_title='Total Count', xaxis_title='Education Level',
                            showlegend=False, title="Educational Level of Targets",
                            height=450, width=450,
                            xaxis_showgrid=False, yaxis_showgrid=False)
            st.write(fig3)

        # How are the income brackets distributed?
        with col4:
            income = pd.Series(df['Income'])
            df['Income_Bracket'] = (np.floor(income / 1000) * 1000).astype('int64')
            fig4 = px.histogram(df, x='Income_Bracket')
            fig4.layout.update(yaxis_title='Total Count', xaxis_title='Income Bracket',
                            showlegend=False, title="Distribution of Income Brackets",
                            height=450, width=450,
                            xaxis_showgrid=False, yaxis_showgrid=False)
            st.write(fig4)

        ## Interpretation of visuals
        st.warning('Key Takeaways:')
        # idx() returns the index/label of the associated value
        st.write('(1) It looks like most of our customers originate from', values.idxmax(), 'with a total count of ',
                values.max(), 'people. In contrast, we should focus on improving our reach with \
                customers from ', values.idxmin(), '(Only ', values.min(), '!).')

        # We will an additional variable to extract max() for the line graph
        resp_v = responded.sort_values('Response', ascending=False)    #Sorts values in descending order for accurate indexing

        st.write('(2) Elsewhere, people born in the following decade were the most receptive to our campaigns:',
                resp_v.iloc[0],                                        #Selects first row, which is sorted as max response rate
                'We might need to look further into the following generation\'s alarmingly low response rate:',
                resp_v.iloc[-1],
                'Possible reasons could be no personalization, wrong messaging for that particular audience, \
                lack of proper testing, or underwhelming calls to action.')

        # Let's calculate the quartiles to determine the IQR
        Q1 = np.percentile(df['Income_Bracket'], 25)
        Q3 = np.percentile(df['Income_Bracket'], 75)

        st.write('(3) In terms of income, it appears that the histogram follows a Normal Distribution, and the majority \
        of user income ranges between ', Q1, 'and ', Q3, 'per year.')

        # One last variable for the fourth graph
        edu_l = df['Education'].unique()
        edu_v = df['Education'].value_counts()

        st.write('(4) Higher education is a recurring theme when it comes to education demographics. Indeed, ',
        edu_v.max(), ' users have a ', edu_v.idxmax(), 'degree, with ', edu_l[1],'(', edu_v[1], ') ' 'and', edu_l[2],
        ' holders in second and third place respectively. Relatively speaking, very few people (', edu_v.min(), ') have ',
        edu_v.idxmin(), ' education.')


## MACHINE LEARNING

# Defining a function for OneHotEncoding that retains feature names (SKlearn)
# Reference: https://github.com/gdiepen/PythonScripts/blob/master/dataframe_onehotencoder.py
class DataFrameOneHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(
        self,
        categories="auto",
        drop=None,
        sparse=None,
        dtype=np.float64,
        handle_unknown="error",
        col_overrule_params={},
    ):

        self.categories = categories
        self.drop = drop
        self.sparse = sparse
        self.dtype = dtype
        self.handle_unknown = handle_unknown
        self.col_overrule_params = col_overrule_params
        pass

    def fit(self, X, y=None):
        if type(X) != pd.DataFrame:
            raise TypeError(f"X should be of type dataframe, not {type(X)}")

        self.onehotencoders_ = []
        self.column_names_ = []

        for c in X.columns:
            # Construct the OHE parameters using the arguments
            ohe_params = {
                "categories": self.categories,
                "drop": self.drop,
                "sparse": False,
                "dtype": self.dtype,
                "handle_unknown": self.handle_unknown,
            }
            # and update it with potential overrule parameters for the current column
            ohe_params.update(self.col_overrule_params.get(c, {}))

            # Regardless of how we got the parameters, make sure we always set the
            # sparsity to False
            ohe_params["sparse"] = False

            # Now create, fit, and store the onehotencoder for current column c
            ohe = OneHotEncoder(**ohe_params)
            self.onehotencoders_.append(ohe.fit(X.loc[:, [c]]))

            # Get the feature names and replace each x0_ with empty and after that
            # surround the categorical value with [] and prefix it with the original
            # column name
            feature_names = ohe.get_feature_names()
            feature_names = [x.replace("x0_", "") for x in feature_names]
            feature_names = [f"{c}[{x}]" for x in feature_names]

            self.column_names_.append(feature_names)

        return self

    def transform(self, X):
        if type(X) != pd.DataFrame:
            raise TypeError(f"X should be of type dataframe, not {type(X)}")

        if not hasattr(self, "onehotencoders_"):
            raise NotFittedError(f"{type(self).__name__} is not fitted")

        all_df = []

        for i, c in enumerate(X.columns):
            ohe = self.onehotencoders_[i]

            transformed_col = ohe.transform(X.loc[:, [c]])

            df_col = pd.DataFrame(transformed_col, columns=self.column_names_[i])
            all_df.append(df_col)

        return pd.concat(all_df, axis=1)

# Let's use the new function for OneHotEncoding
# Reference: https://www.guidodiepen.nl/2021/02/keeping-column-names-when-using-sklearn-onehotencoder-on-pandas-dataframe/
df2 = df[['Education', 'Marital_Status', 'Country']]
df_ohe = DataFrameOneHotEncoder()
ohe_done = pd.DataFrame(df_ohe.fit_transform(df2))

# Merge both back into one dataframe
df.reset_index(inplace=True)
df = pd.concat([df, ohe_done], axis=1)
# Drop old columns
df = df.drop(['Education', 'Marital_Status', 'Country'], axis=1)
df.reset_index(inplace=True)
df.drop('index', axis=1, inplace=True)
df.drop('level_0', axis=1, inplace=True)

# Split the dataset into X_train and y_train to determine features vs target variable
X = df.drop('Response', axis=1)
y = df['Response']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1131)

# ML Model Selection
# So what's the model performance? Let's try Linear Regression and calculate RMSE
lr = LinearRegression()
lr.fit(X_train, y_train.values.ravel())

lr_tr_predictions = lr.predict(X_train)
lr_train_rmse = mean_squared_error(y_train, lr_tr_predictions, squared=False)

lr_ts_predictions = lr.predict(X_test)
lr_test_rmse = mean_squared_error(y_test, lr_ts_predictions, squared=False)

# What's the mean RMSE if we perform 5-Fold Cross-Validation?
lr_scores = (cross_val_score(lr, X_train, y_train.values.ravel(), scoring='neg_root_mean_squared_error', cv=5))
lr_scores.mean()
lr_scores.std()

# Maybe a Decision Tree algorithm would have better RMSE?
dtr = DecisionTreeRegressor(random_state=42)
dtr.fit(X_train, y_train.values.ravel())

dtr_tr_predictions = dtr.predict(X_train)
dtr_train_rmse = mean_squared_error(y_train, dtr_tr_predictions, squared=False)

dtr_ts_predictions = dtr.predict(X_test)
dtr_test_rmse = mean_squared_error(y_test, dtr_ts_predictions, squared=False)

tree_scores = (cross_val_score(dtr, X_train, y_train.values.ravel(), scoring='neg_root_mean_squared_error', cv=5))
print(tree_scores)
tree_scores.mean()
tree_scores.std()               # The actual RMSE is simply the positive version of the number you're getting.

# Random Forest Regressor
rfr = RandomForestRegressor(n_estimators=10, random_state=1111)
rfr.fit(X_train, y_train.values.ravel())

rfr_tr_predictions = rfr.predict(X_train)
rfr_train_rmse = mean_squared_error(y_train, rfr_tr_predictions, squared=False)

rfr_ts_predictions = rfr.predict(X_test)
rfr_test_rmse = mean_squared_error(y_test, rfr_ts_predictions, squared=False)

forest_scores = (cross_val_score(rfr, X_train, y_train.values.ravel(), scoring='neg_root_mean_squared_error', cv=5))
print(forest_scores)
forest_scores.mean()
forest_scores.std()

# Gradient Boosting Regressor
gbr = GradientBoostingRegressor(random_state=1111)
gbr.fit(X_train, y_train.values.ravel())

gbr_tr_predictions = gbr.predict(X_train)
gbr_train_rmse = mean_squared_error(y_train, gbr_tr_predictions, squared=False)

gbr_ts_predictions = gbr.predict(X_test)
gbr_test_rmse = mean_squared_error(y_test, gbr_ts_predictions, squared=False)

boost_scores = (cross_val_score(gbr, X_train, y_train.values.ravel(), scoring='neg_root_mean_squared_error', cv=5))
print(boost_scores)
boost_scores.mean()
boost_scores.std()

# Create Table to compare Model Performance
compare = pd.DataFrame([['Decision Tree', dtr_train_rmse, dtr_test_rmse, tree_scores.mean()],
              ['Random Forest Regressor', rfr_train_rmse, rfr_test_rmse, forest_scores.mean()],
              ['Linear Regression', lr_train_rmse, lr_test_rmse, lr_scores.mean()],
              ['Gradient Boosting Regressor', gbr_train_rmse, gbr_test_rmse, boost_scores.mean()]],
             columns=['Model', 'Train RMSE', 'Test RMSE','Cross-Val RMSE'],).T     #.T = Flips the X and Y of the table

# Gradient Boosting Regressor is the champion! Let's select it for the next step
# What's the predictive power of GBR?
y_predict = gbr.predict(X_test)

# Convert it into an array
y_pred= []
for element in y_predict:
  if element > 0.5:
    y_pred.append(1)
  else:
    y_pred.append(0)

# Calculate classification report
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Compute and visualize confusion matrix
ConfusionMatrix = confusion_matrix(y_test, y_pred)

sns.set()
plt.rcParams['figure.figsize'] = [3,3]
fig, ax= plt.subplots()
ax = sns.heatmap(ConfusionMatrix, annot=True, fmt='d')
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

# Time to design the ML section on the app
if option == "Machine Learning":
    from PIL import Image
    image = Image.open('line.png')    #Image of a line to separate new section from overview
    st.image(image)

    st.write("Machine Learning Magic! The transformed dataset has: ", df.shape[0], " records and ", df.shape[1], " features")

    st.warning("Let's dive into model performance and evaluation")

    st.write('Which model returned the best performance?', compare)
    scores_l = {'Gradient Boosting Regressor':boost_scores.mean() , 'Random Forest Regressor':forest_scores.mean(),
                'Decision Tree Regressor':tree_scores.mean(), 'Linear Regression':lr_scores.mean()}
    st.write('By the looks of things, it is recommended to select ', max(scores_l),
                'with an average RMSE of ', scores_l.get(max(scores_l)))

    col5, col6= st.beta_columns([2,3])
    with col5:
        st.pyplot(fig)
    with col6:
        # Assess feature importance according to ML
        ranking = np.argsort(-rfr.feature_importances_)
        sns.set()
        plt.rcParams['figure.figsize'] = [8,8]
        fig, ax = plt.subplots()
        ax = sns.barplot(x=rfr.feature_importances_[ranking], y=X_train.columns.values[ranking], orient='h')
        ax.set_xlabel("Feature Importance")
        plt.tight_layout()
        st.pyplot(fig)
