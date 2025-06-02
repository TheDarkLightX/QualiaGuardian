# This file is for testing hardcoded secrets detection.

API_KEY = "sk_live_12345abcDEfgHiJkLMnOpQrStUvWxYz"
DATABASE_PASSWORD = 'mySup3rS3cur3P@ssw0rd!'
some_other_variable = "this is fine"
# A comment: SECRET_KEY = "commented_out_secret"
another_api_token = "ghp_abcdefghijklmnopqrstuvwxyz123456" # Generic token
short_key = "123" # Should be ignored by generic pattern if too short
empty_secret = "" # Should be ignored
null_secret = None # Not a string literal

class Config:
    SECRET_CONFIG_VALUE = "config_secret_value_example"
    master_key = 'master_of_keys_123'

# Example of a dictionary that might be flagged by broader patterns
sensitive_data = {
    "user_password": "userpass123",
    "service_token": "service_token_value_abc"
}