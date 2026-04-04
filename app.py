# -----------------------------
# 1. Import Libraries
# -----------------------------
import streamlit as st
import pandas as pd
from datetime import date
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -----------------------------
# 2. Load Dataset
# -----------------------------
petrol = pd.read_csv("petrol.csv")
diesel = pd.read_csv("diesel.csv")

# Standardize column names
petrol.columns = petrol.columns.str.lower()
diesel.columns = diesel.columns.str.lower()

# Detect columns
def find_columns(df):
    date_col, price_col, state_col = None, None, None
    for col in df.columns:
        if 'date' in col:
            date_col = col
        if 'price' in col or 'rs' in col or 'rate' in col:
            price_col = col
        if 'state' in col or 'location' in col:
            state_col = col
    return date_col, price_col, state_col

p_date, p_price, p_state = find_columns(petrol)
d_date, d_price, d_state = find_columns(diesel)

# Rename columns
petrol = petrol.rename(columns={p_date:'date', p_price:'petrol_price', p_state:'state'})
diesel = diesel.rename(columns={d_date:'date', d_price:'diesel_price', d_state:'state'})

# Convert date
petrol['date'] = pd.to_datetime(petrol['date'], errors='coerce')
diesel['date'] = pd.to_datetime(diesel['date'], errors='coerce')

# Drop missing
petrol = petrol.dropna(subset=['date', 'petrol_price', 'state'])
diesel = diesel.dropna(subset=['date', 'diesel_price', 'state'])

# -----------------------------
# 3. Streamlit UI
# -----------------------------
st.title("🟢 Smart Fuel Price Prediction")
st.write("Predict Petrol & Diesel Price (₹ per litre) for a selected state and date.")

# Select state
available_states = sorted(petrol['state'].unique())
selected_state = st.selectbox("Select State", available_states)

# Select date
input_date = st.date_input("Select Date", date.today())

# -----------------------------
# 4. Filter Data by State
# -----------------------------
petrol_state = petrol[petrol['state'] == selected_state].copy()
diesel_state = diesel[diesel['state'] == selected_state].copy()

# Feature engineering
for df in [petrol_state, diesel_state]:
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

# -----------------------------
# 5. Train ML Models
# -----------------------------
petrol_model = LinearRegression()
petrol_model.fit(petrol_state[['year','month','day']], petrol_state['petrol_price'])

diesel_model = LinearRegression()
diesel_model.fit(diesel_state[['year','month','day']], diesel_state['diesel_price'])

# -----------------------------
# 6. Prediction
# -----------------------------
year, month, day = input_date.year, input_date.month, input_date.day

petrol_price = petrol_model.predict([[year, month, day]])[0]
diesel_price = diesel_model.predict([[year, month, day]])[0]

# Show results (₹ per litre)
st.subheader("🔮 Predicted Prices")

st.success(f"⛽ Petrol Price in {selected_state} on {input_date}: ₹{petrol_price:.2f} per litre")
st.success(f"🚗 Diesel Price in {selected_state} on {input_date}: ₹{diesel_price:.2f} per litre")

# -----------------------------
# 7. Show Dataset Tables
# -----------------------------
st.subheader("📊 Filtered Dataset (Selected State)")

if st.checkbox("Show Petrol Dataset"):
    st.write("Petrol Data")
    st.dataframe(petrol_state)

if st.checkbox("Show Diesel Dataset"):
    st.write("Diesel Data")
    st.dataframe(diesel_state)

# -----------------------------
# 8. Charts
# -----------------------------
st.subheader(f"📈 Petrol Price Trend - {selected_state}")
fig1, ax1 = plt.subplots()
ax1.plot(petrol_state['date'], petrol_state['petrol_price'])
ax1.set_xlabel("Date")
ax1.set_ylabel("Price (₹ per litre)")
st.pyplot(fig1)

st.subheader(f"📈 Diesel Price Trend - {selected_state}")
fig2, ax2 = plt.subplots()
ax2.plot(diesel_state['date'], diesel_state['diesel_price'])
ax2.set_xlabel("Date")
ax2.set_ylabel("Price (₹ per litre)")
st.pyplot(fig2)