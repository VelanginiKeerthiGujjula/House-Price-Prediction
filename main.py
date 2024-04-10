from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)
data = pd.read_csv('final_dataset.csv')
pipe = pickle.load(open("RidgeModel.pkl", 'rb'))

# Define preprocessing steps
categorical_features = ['zip_code']  # Specify categorical features
numeric_features = ['beds', 'baths', 'size']  # Specify numeric features

# Handle unknown categories in the input data
def preprocess_input(input_data):
    for column in input_data.columns:
        unknown_categories = set(input_data[column]) - set(data[column].unique())
        if unknown_categories:
            # Handle unknown categories (e.g., replace with a default value)
            input_data[column] = input_data[column].replace(unknown_categories, data[column].mode()[0])
    return input_data

@app.route('/')
def index():
    bedrooms = sorted(data['beds'].unique())
    bathrooms = sorted(data['baths'].unique())
    sizes = sorted(data['size'].unique())
    zip_codes = sorted(data['zip_code'].unique())

    return render_template('index.html', bedrooms=bedrooms, bathrooms=bathrooms, sizes=sizes, zip_codes=zip_codes)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        bedrooms = request.form.get('beds')
        bathrooms = request.form.get('baths')
        size = request.form.get('size')
        zipcode = request.form.get('zip_code')

        # Convert the input data to the expected format
        input_data = pd.DataFrame({'beds': [bedrooms],
                                   'baths': [bathrooms],
                                   'size': [size],
                                   'zip_code': [zipcode]})

        # Preprocess the input data
        processed_input_data = preprocess_input(input_data)

        # Predict the price
        prediction = pipe.predict(processed_input_data)[0]
        return f"Price: INR {prediction}"  # Assuming prediction is in INR
    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True, port=5000)
