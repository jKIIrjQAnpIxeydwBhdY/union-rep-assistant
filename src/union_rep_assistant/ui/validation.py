import requests
import streamlit as st


def is_valid_api_key(api_key):
    url = "https://api.openai.com/v1/models"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            return True
        elif response.status_code == 401:
            st.info("Invalid API key!")
            return False
        else:
            st.info(f"Unexpected error: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        st.info(f"Error checking API key: {e}")
        return False
