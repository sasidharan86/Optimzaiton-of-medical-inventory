# Import necessary libraries
import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("model.pickle", "rb"))

# Create a function for prediction
def predict_quantity(quantity_last_month, quantity_2_months_back, quantity_3_months_back):
    input_data = np.array([quantity_last_month, quantity_2_months_back, quantity_3_months_back]).reshape(1, -1)
    prediction = model.predict(input_data)
    return prediction[0]

# Streamlit app
def main():
    st.title("Medical Inventory Quantity Prediction")
    st.sidebar.header("User Input")

    # User input for quantity values
    quantity_last_month = st.sidebar.number_input("Quantity Last Month", min_value=0)
    quantity_2_months_back = st.sidebar.number_input("Quantity 2 Months Back", min_value=0)
    quantity_3_months_back = st.sidebar.number_input("Quantity 3 Months Back", min_value=0)

    # Make predictions
    if st.sidebar.button("Predict"):
        prediction = predict_quantity(quantity_last_month, quantity_2_months_back, quantity_3_months_back)
        st.sidebar.success(f"Predicted Quantity: {prediction:.2f}")

    # Display raw data
    st.subheader("Raw Data")
    st.write("You can view the raw data if needed.")
    st.write(df_grouped)

if __name__ == "__main__":
    main()

