from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load the model
pipe = pickle.load(open('pipe.pkl', 'rb'))

teams = [
    'Australia', 'India', 'Bangladesh', 'New Zealand', 'South Africa', 
    'England', 'West Indies', 'Afghanistan', 'Pakistan', 'Sri Lanka'
]

cities = [
    'Colombo', 'Mirpur', 'Johannesburg', 'Dubai', 'Auckland', 'Cape Town', 
    'London', 'Pallekele', 'Barbados', 'Sydney', 'Melbourne', 'Durban', 
    'St Lucia', 'Wellington', 'Lauderhill', 'Hamilton', 'Centurion', 
    'Manchester', 'Abu Dhabi', 'Mumbai', 'Nottingham', 'Southampton', 
    'Mount Maunganui', 'Chittagong', 'Kolkata', 'Lahore', 'Delhi', 
    'Nagpur', 'Chandigarh', 'Adelaide', 'Bangalore', 'St Kitts', 'Cardiff', 
    'Christchurch', 'Trinidad'
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Retrieve input values safely
            batting_team = request.form.get('batting_team', '')
            bowling_team = request.form.get('bowling_team', '')
            city = request.form.get('city', '')
            current_score = int(request.form.get('current_score', 0))
            overs_done = float(request.form.get('overs_done', 0))
            wickets = int(request.form.get('wickets', 0))
            last_five = int(request.form.get('last_five', 0))

            # Compute required features
            balls_left = 120 - (overs_done * 6)
            wicket_left = 10 - wickets  # ✅ Corrected name
            current_run_rate = current_score / overs_done if overs_done else 0  # ✅ Corrected name

            # Create DataFrame with correct column names
            input_df = pd.DataFrame({
                'batting_team': [batting_team],
                'bowling_team': [bowling_team],
                'city': [city],
                'current_score': [current_score],
                'balls_left': [balls_left],
                'wicket_left': [wicket_left],  # ✅ Matches model expectation
                'current_run_rate': [current_run_rate],  # ✅ Matches model expectation
                'last_five': [last_five]
            })

            # Predict using the model
            result = pipe.predict(input_df)
            prediction = int(result[0])

        except Exception as e:
            print(f"Error: {e}")  # Debugging output

    return render_template('index.html', teams=sorted(teams), cities=sorted(cities), prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
