
# Predicting Bestseller Novels on Amazon.com


## Project Overview
This project aims to predict whether a novel listed on Amazon will become a bestseller based on specific attributes. 
By analyzing data collected from Amazon's "novels" search results, the project leverages machine learning techniques to build and deploy a predictive model. 
This is an end-to-end machine learning project, covering all stages from data collection to deployment in AWS' EC2.
It involves: <br>
- data collection through web scraping (building the web scraper from scratch) <br>
- data cleaning and transformation <br>
- data exploration <br>
- data analysis <br>
- training a predictive ML model on the data to predict whether a novel is a bestseller - see the jupyter notebook <br>
- deploying the model within a flask web application on my local host  - see the app.py file <br>
- deploying the flask web application to the cloud using AWS' EC2

## Technologies Used
- **Python**: The core programming language used for data collection, analysis, and model building.
- **BeautifulSoup**: A Python library used for web scraping and data collection from Amazon.
- **Pandas & NumPy**: Libraries used for data manipulation and transformation.
- **Scikit-learn**: A machine learning library used for model training and evaluation.
- **Flask**: A lightweight web framework used to deploy the predictive model as a web application.
- **AWS EC2**: Amazon Web Services Elastic Compute Cloud, used for deploying the Flask web application to the cloud.

## Installation and Local Usage
1. **Clone the Repository**:
   - git clone "https://github.com/mngembu/Bestseller-Predictor-Deployed-to-AWS-EC2.git"
   - cd Bestseller-Predictor-Deployed-to-AWS-EC2

2. **Set Up the Virtual Environment** 
- python -m venv venv
- `venv\Scripts\activate`

3. **Install Dependencies**:
- python -m pip install -r requirements.txt

4. **Run the Flask App**
- python app.py

5. **Open App in your browser**:
- Access the web application at http://127.0.0.1:5000/ in your browser.

## Dependencies
- BeautifulSoup4: Used for web scraping and data extraction from Amazon.
- pandas & NumPy: Libraries for data manipulation and transformation.
- scikit-learn: Used for building and evaluating the machine learning model.
- Flask: For deploying the model as a web application.
- Seaborn
- Matplotlib


## Screenshots

![image](https://user-images.githubusercontent.com/56229226/194883632-ff211c29-585c-4669-8ae8-fb77d01d7170.png)

![image](https://user-images.githubusercontent.com/56229226/194883879-54d2d8cb-51e4-47f6-9006-0794f6d45332.png)



## Contact

If you have any questions, feel free to reach out to me at ara.ngembu@yahoo.com.

Author: Mary Ara Ngembu





