# Wine Quality Prediction ML Project

## Overview
This project implements an end-to-end machine learning pipeline for predicting wine quality based on physicochemical properties. It includes data ingestion, validation, transformation, model training, and evaluation stages.

## Project Structure
```
end-to-end-project-wine-quality/
│
├── .github/
│   └── workflows/
│       └── github.yaml
├── src/
│   └── mlProject/
│       ├── components/
│       │   ├── data_ingestion.py
│       │   ├── data_validation.py
│       │   ├── data_transformation.py
│       │   ├── model_training.py
│       │   └── model_evaluation.py
│       ├── pipeline/
│       │   └── training_pipeline.py
│       ├── utils/
│       │   └── common.py
│       ├── config/
│       │   └── configuration.py
│       ├── entity/
│       │   └── config_entity.py
│       ├── __init__.py
│       └── exception.py
├── config/
│   └── config.yaml
├── params.yaml
├── schema.yaml
├── templates/
│   └── index.html
├── app.py
├── main.py
├── setup.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/end-to-end-project-wine-quality.git
cd end-to-end-project-wine-quality
```

2. Create and activate virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Update configuration in `config.yaml` and parameters in `params.yaml`

2. Run the training pipeline:
```bash
python main.py
```

3. Start the web application:
```bash
python app.py
```

4. Access the prediction interface at `http://localhost:8080`

## Project Components

### Data Pipeline
- **Data Ingestion**: Loads and prepares raw data
- **Data Validation**: Validates data schema and quality
- **Data Transformation**: Preprocesses data for model training

### Model Pipeline
- **Model Training**: Trains ElasticNet regression model
- **Model Evaluation**: Evaluates model performance
- **Model Serving**: Flask web application for predictions

## Configuration

### config.yaml
Contains configurations for:
- Data paths
- Model parameters
- Pipeline settings

### params.yaml
Contains model hyperparameters:
- ElasticNet parameters (alpha, l1_ratio)
- Training parameters (test_size, random_state)

## Model Performance
- R2 Score: 0.XX
- MAE: 0.XX
- RMSE: 0.XX

## CI/CD Pipeline
GitHub Actions workflow includes:
- Automated testing
- Code quality checks
- Deployment pipeline

## Technologies Used
- Python 3.10+
- scikit-learn
- Flask
- pandas
- numpy
- pytest
- GitHub Actions

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
Your Name - your.email@example.com
Project Link: https://github.com/yourusername/end-to-end-project-wine-quality