# Import python packages
import streamlit as st
import json
from snowflake.snowpark.context import get_active_session

# Snowpark for Python
from snowflake.snowpark import Session
from snowflake.snowpark.version import VERSION
from snowflake.snowpark.functions import udf
import snowflake.snowpark.functions as F

# Snowpark ML
from snowflake.ml.modeling.xgboost import XGBRegressor
from snowflake.ml.modeling.model_selection import GridSearchCV
from snowflake.ml.registry import Registry
from snowflake.ml._internal.utils import identifier


# Write directly to the app
st.title("Diamond Price Prediction :ring:")
st.write(
    """Call your newly deployed diamond prediction model from Snowflake's model registry!
    """
)

# Set up the session
connection_parameters = json.load(open('connection.json'))
session = Session.builder.configs(connection_parameters).create()
session.sql_simplifier_enabled = True

snowflake_environment = session.sql('SELECT current_user(), current_version()').collect()
snowpark_version = VERSION

# Current Environment Details
print('\nConnection Established with the following parameters:')
print('User                        : {}'.format(snowflake_environment[0][0]))
print('Role                        : {}'.format(session.get_current_role()))
print('Database                    : {}'.format(session.get_current_database()))
print('Schema                      : {}'.format(session.get_current_schema()))
print('Warehouse                   : {}'.format(session.get_current_warehouse()))
print('Snowflake version           : {}'.format(snowflake_environment[0][1]))
print('Snowpark for Python version : {}.{}.{}'.format(snowpark_version[0],snowpark_version[1],snowpark_version[2]))
db = identifier._get_unescaped_name(session.get_current_database())
schema = identifier._get_unescaped_name(session.get_current_schema())

#Check out available model versions
st.write("ML Model Version Table")
model_name = "DIAMONDS_PRICE_PREDICTION"
native_registry = Registry(session=session, database_name=db, schema_name=schema)
st.write(native_registry.get_model(model_name).show_versions())


#Create fields to specify model inputs
st.write("Model inputs")

#Model input columns
input_cols =[
 'CARAT',
 'DEPTH',
 'TABLE_PCT',
 'X',
 'Y',
 'Z',
 'CUT_OE',
 'COLOR_OE',
 'CLARITY_OE',
 'COLOR',
 'CLARITY',
 'CUT']

#Column data types
input_data_types = ['float',
 'float',
 'float',
 'float',
 'float',
 'float',
 'float',
 'float',
 'float',
 'string',
 'string',
 'string']

#Default values
input_default_vals = [0.25,
 0.5,
 0.30,
 0.4,
 0.075,
 0.075,
 0.0,
 3.0,
 4.0,
 'G',
 'VS2',
 'IDEAL']

#initialize dictionary for storing inputs
prediction_input_dict = dict()

#Loop through columns, add a streamlit entry field for the appropriate column type
#Take the input data to create a table we will then feed into our model
for n,col in enumerate(input_cols):
    if input_data_types[n]=="float":
        input_val = st.number_input(
            f"{col}",
            value=input_default_vals[n],
            help=f"enter the value for {col}",
            format="%.2f")
        prediction_input_dict.update({col:input_val})
    if input_data_types[n]=="string":
        input_val = st.text_input(
            f"{col}",
            value=input_default_vals[n],
            help=f"enter the value for {col}")
        prediction_input_dict.update({col:input_val})

#Create a dataframe
input_df = session.create_dataframe([prediction_input_dict])

#Get actual model (version 1) and run predict against input data
#Write to a table and show results on the dashboard!
model_ver = native_registry.get_model(model_name).version('v1')
result = model_ver.run(input_df, function_name="predict")
result = result.rename({result.columns[-1]: "predicted_price"})

#Select, round, format, and display the final result 
predicted_price = f"{round(result.select('predicted_price').collect()[0].PREDICTED_PRICE,2):,}"
st.markdown("#")
st.subheader(f"Diamond price prediction - ${predicted_price}")

st.markdown("#")
st.write("Full data input and output")
st.write(result)