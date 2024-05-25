import streamlit as st # pip install streamlit
import pandas as pd
import numpy as np

st.title('Simple Streamlit Dashboard')

st.write('Welcome to your simple dashboard!')

# Create a random dataframe
df = pd.DataFrame(
    np.random.randn(10, 5),
    columns=('col %d' % i for i in range(5)))

st.write('Here is a random dataframe:')
st.write(df)

# Line chart
st.line_chart(df)

# User input
user_input = st.text_input("Enter a number", 0)
st.write(f'You entered: {user_input}')
