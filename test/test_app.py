import pytest
from scripts.apps.app import ma_fonction_streamlit # Import the functions you want to test

def test_ma_fonction_streamlit():
    # Define test inputs
    input_value = "test"
    # Call the Streamlit function
    result = ma_fonction_streamlit(input_value)
    # Check the expected result
    assert result == "Expected Result" # Replace with the correct result