# RCC-Survival-Predictor
RCC Survival Predictor is a web application designed to assist oncologists and healthcare professionals in estimating overall survival rates for patients with Renal Cell Carcinoma (RCC). By leveraging machine learning algorithms, the app provides accurate predictions based on a comprehensive set of clinical variables specific to RCC.


https://github.com/user-attachments/assets/a2681e49-3029-423c-a30d-387071252d08




# RCC Survival Predictor

## Description
RCC Survival Predictor is a web application designed to assist oncologists and healthcare professionals in estimating overall survival rates for patients with Renal Cell Carcinoma (RCC). By leveraging machine learning algorithms, the app provides accurate predictions based on a comprehensive set of clinical variables specific to RCC.

## Features
- Predict overall survival (OS) in months for RCC patients
- User-friendly interface for inputting patient data
- Visualization of prediction results
- Dark/Light mode for comfortable viewing
- Responsive design for various devices

## Tech Stack
- Backend: Python, Flask
- Frontend: HTML, CSS, JavaScript
- ML Model: Scikit-learn (Random Forest and Gradient Boosting)
- Additional libraries: Pandas, NumPy, Matplotlib

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/your-username/rcc-survival-predictor.git
   ```
2. Navigate to the project directory:
   ```
   cd RCC-survival-predictor
   ```
3. Create a virtual environment:
   ```
   python -m venv venv
   ```
4. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS and Linux: `source venv/bin/activate`
5. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage
1. Start the Flask server:
   ```
   python app.py
   ```
2. Open a web browser and go to `http://localhost:5000`
3. Use the interface to input patient data and receive survival predictions

## Model Information
The prediction model uses a combination of Random Forest and gradient-boosting techniques trained on historical RCC patient data. The model takes into account various clinical variables such as age, sex, tumour grade, and treatment modalities to provide an estimated overall survival in months.

## Contributing
Contributions to improve the RCC Survival Predictor are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature branch)
3. Make your changes and commit them (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin feature branch)
5. Create a new Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact
For any queries or suggestions, please contact us at contact@rccpredictor.com.

## Acknowledgments
- Thanks to all contributors who have helped in developing this tool
- Special thanks to the oncology departments who provided valuable insights and data for model training
