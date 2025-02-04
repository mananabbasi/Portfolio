
# 🎯 **Data Scientist Portfolio**  
🌟 Welcome to my **Data Scientist Portfolio** repository! 🚀  
This is a curated collection of my work, showcasing my expertise in **data analysis, machine learning, and visualization**.  
Each project reflects my ability to extract meaningful insights, develop predictive models, and present data-driven solutions effectively.

---

## 📌 **Table of Contents**  
- [🌟 About Me](#-about-me)  
- [🚀 Projects](#-projects)  
- [🛠 Skills](#-skills)  
- [⚙ Tools & Technologies](#-tools--technologies)  
- [💼 Work Experience](#-work-experience)  
- [🎓 Education](#-education)  
- [🎯 Activities](#-activities)  
- [🔧 How to Use This Repository](#-how-to-use-this-repository)  
- [📞 Contact](#-contact)  

---

## 🌟 About Me  
<a id="-about-me"></a>  
I am a **data scientist** with a passion for solving complex problems using data-driven approaches. I specialize in transforming raw data into actionable insights to drive informed decision-making.  
My expertise includes **statistics**, **machine learning**, **data visualization**, and **big data processing**. I am constantly learning and adapting to new technologies to improve my analytical skills and broaden my knowledge in the data science field.

### 📊 **Key Expertise:**  
- **Data Wrangling & Cleaning**  
- **Exploratory Data Analysis (EDA)**  
- **Machine Learning & Predictive Modeling**  
- **Statistical Analysis & Hypothesis Testing**  
- **Data Visualization & Storytelling**  

---


## 🚀 Projects

---

### 🔹 **1. Developed Hospital Databases from Scratch using Microsoft SQL Server**  
**Duration:** Jan 2024 - Apr 2024

**Objective:**  
To design and implement a scalable relational Hospital database system capable of efficiently managing large volumes of structured data. The system needed to be optimized for performance, with clear data integrity and fast query execution.

**Process:**  
- **Database Design**: Started with analyzing the business requirements to determine the essential entities and their relationships. Created an **Entity-Relationship (ER) Diagram** to visualize the schema.  
- **Normalization**: Applied normalization techniques (up to 3rd Normal Form) to ensure data consistency, reduce redundancy, and improve maintainability.  
- **Table Creation**: Developed tables with primary keys and foreign keys to define relationships between the entities.  
- **Indexing**: Implemented indexing on frequently queried columns to enhance query performance.  
- **Stored Procedures & Triggers**: Wrote stored procedures to automate repetitive tasks like data insertion, updates, and deletions. Triggers were used for enforcing business rules, such as auditing or restricting certain actions.  
- **Performance Optimization**: Used query optimization techniques such as indexing, query refactoring, and minimizing joins to ensure high performance.

**Outcome:**  
- A fully operational relational database system that efficiently manages business data.
- Optimized for high-speed query execution even with large datasets.
- Scalable design that can handle future growth.

**Tools Used:**  
- **Microsoft SQL Server**  
- **T-SQL**

Below are some screenshots from my database, including ER Diagrams, Views, Triggers, and Stored Procedures:

### Database Diagram
![Database Diagram](assets/Data_Base_Diagram.png)

### Total Appointments
![Appointments](assets/Appointsments.png)

### Returning Today's Appointments
![Returning Today's Appointments](assets/Returing_Todays Apointsments.png)

### Updating Doctor Table
![Doctor Update](assets/Doctor_Update.png)

### 🔹 **2. Conducted Statistical Analysis & Advanced Statistics using R**  
**Duration:** Sep 2024 - Dec 2025  
**Dataset:** [Concrete Strength Prediction](https://www.kaggle.com/datasets/ruchikakumbhar/concrete-strength-prediction)  

**Objective:**  
To perform comprehensive statistical analysis on the dataset to uncover insights, identify trends, and support data-driven decision-making. The goal was to understand relationships between variables and provide actionable recommendations to stakeholders.

**Process:**  
1. **Data Cleaning:**  
   - Handled missing values using imputation techniques.  
   - Identified and removed outliers using statistical methods (e.g., IQR, Z-score).  
   - Ensured data consistency by standardizing formats and correcting errors.  

2. **Exploratory Data Analysis (EDA):**  
   - Analyzed variable distributions using histograms, box plots, and density plots.  
   - Explored relationships between variables using scatter plots and correlation matrices.  
   - Identified patterns and trends in the data.  

3. **Hypothesis Testing:**  
   - Conducted t-tests and chi-square tests to validate assumptions and test hypotheses.  
   - Determined statistical significance of findings.  

4. **Regression Analysis:**  
   - Built linear regression models to quantify relationships between independent and dependent variables.  
   - Evaluated model performance using metrics like R-squared and RMSE.  

5. **ANOVA:**  
   - Performed Analysis of Variance (ANOVA) to compare means across groups and identify significant differences.  

6. **Visualization:**  
   - Created clear and informative visualizations using **ggplot2** to communicate insights effectively.  
   - Generated plots such as histograms, scatter plots, and correlation heatmaps.  

**Outcome:**  
- Delivered actionable insights to stakeholders, enabling data-driven decision-making.  
- Provided detailed statistical reports summarizing key findings and trends.  
- Created visualizations and dashboards to present findings in an accessible format.  

**Tools Used:**  
- **R** (ggplot2, dplyr, tidyr, caret, stats)  

### Key Visualizations:
1. **Libraries Used**  
   ![Libraries Used](assets/Libraries_used.png)  
   - Highlighted the R libraries utilized for analysis.

2. **Numerical Variable Distribution**  
   ![Numerical Distribution](assets/Distribution_of_Numerical.png)  
   - Analyzed and visualized the distribution of numerical variables.

3. **Categorical Variable Distribution**  
   ![Categorical Distribution](assets/Categorical_Distribution.png)  
   - Explored the distribution of categorical variables.

4. **Correlation Analysis**  
   ![Correlation Matrix](assets/Corelation.png)  
   - Created a correlation matrix to identify relationships between variables.

5. **Age Distribution**  
   ![Age Distribution](assets/Distribtion_off_Age.png)  
   - Visualized the distribution of age-related variables.

6. **Simple Linear Regression (SLR) Assumptions**  
   ![SLR Assumptions](assets/SRL_Assumptions.png)  
   - Validated assumptions for regression models.

7. **Regression Model Results**  
   ![Regression Model](assets/RegressionModel.png)  
   - Displayed results and performance metrics of regression models.

8. **Generalized Additive Model (GAM)**  
   ![GAM Model](assets/Gam_Model.png)  
   - Visualized GAM results to capture non-linear relationships.

  
### 🔹 **3. Built Time Series Forecasting Models in R**  
**Duration:** Nov 2024 - Dec 2024  
**Dataset:** [Vital Statistics in the UK](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/vitalstatisticspopulationandhealthreferencetables)  

**Objective:**  
To develop accurate time series forecasting models for predicting future trends, enabling stakeholders to optimize business strategies.

**Process:**  
1. **Data Preparation**:  
   - Collected and cleaned historical time series data.  
   - Ensured stationarity using transformations (e.g., differencing).  

2. **Model Selection**:  
   - Evaluated **ARIMA**, **SARIMA**, and **ETS** models for accuracy.  

3. **Model Evaluation**:  
   - Used **RMSE** and **MAE** to measure model performance.  
   - Performed cross-validation to ensure robustness.  

4. **Visualization**:  
   - Created visualizations to compare predicted vs actual values.  

**Outcome:**  
- Delivered accurate forecasts for future trends.  
- Optimized inventory management and decision-making processes.  

**Tools Used:**  
- **R** (forecast, tseries, ggplot2)  

**Key Visualizations:**  
1. **Time Series Plot**  
   ![Time Series](assets/TimeSeries.png)  
2. **Forecasting Results**  
   ![Forecasting](assets/TS_Forcasting.png)  
3. **Forecast Errors**  
   ![Forecast Errors](assets/TS_Forcast_Errors.png)  
4. **Additive Model with Trend**  
   ![Additive Model](assets/Additive_model_with_increasing_or_decreasing_trend_and_no_seasonality.png)  
---

### 🔹 **4. Designed Power BI Dashboards for Real-Time Insights**  
**Duration:** Jan 2024 - Current

**Objective:**  
To create dynamic and interactive dashboards that allow users to monitor and analyze business performance in real time, helping stakeholders track KPIs and make timely decisions.

**Process:**  
- **Data Integration**: Integrated multiple data sources, including SQL databases, Excel files, and other APIs, into Power BI to ensure data consistency and real-time updates.  
- **Dashboard Design**: Designed user-friendly and visually appealing dashboards that presented key business metrics. Added interactive elements such as slicers and filters to allow users to explore the data from different perspectives.  
- **Advanced Analytics**: Used **DAX (Data Analysis Expressions)** to create calculated measures, columns, and KPIs, enabling more advanced analysis directly within the dashboard.  
- **Performance Optimization**: Ensured that the dashboards were optimized for performance, enabling quick load times even when dealing with large datasets.  
- **Real-Time Data Updates**: Configured Power BI to connect to live data sources, providing real-time insights to the stakeholders.

**Outcome:**  
- Delivered interactive dashboards that allowed stakeholders to monitor performance in real-time.
- Improved decision-making by providing up-to-date, easily interpretable data visualizations.

**Tools Used:**  
- **Power BI**  
- **DAX**, **SQL**

---

### 🔹 **5. Developed Classification & Clustering Models in Python**  
**Duration:** Sep 2024 - Current

**Objective:**  
To build machine learning models for classification and clustering tasks that could predict outcomes and uncover hidden patterns in data.

**Process:**  
- **Data Preprocessing**: Cleaned and preprocessed the data, handling missing values, scaling features, and encoding categorical variables using tools like **Pandas** and **Scikit-learn**.  
- **Model Development**: Built **classification models** such as **Logistic Regression**, **Random Forest**, and **SVM** for binary and multi-class prediction tasks.  
- **Clustering Models**: Implemented **K-Means** and **DBSCAN** for clustering analysis to identify groups of similar data points without predefined labels.  
- **Model Evaluation**: Evaluated classification models using metrics like **accuracy**, **precision**, **recall**, and **F1 score**. Clustering performance was evaluated using the **silhouette score** and other internal validation techniques.  
- **Model Tuning**: Used **GridSearchCV** and **RandomizedSearchCV** to fine-tune model hyperparameters for optimal performance.

**Outcome:**  
- Developed robust machine learning models that provided accurate predictions for various business use cases.
- Identified hidden patterns and clusters within the data, providing valuable insights into customer behavior and other business areas.

**Tools Used:**  
- **Python**  
- **Scikit-learn**, **Pandas**, **Matplotlib**

---

### 🔹 **6. Executed PySpark Projects on Databricks Using RDDs & DataFrames**  
**Duration:** Jan 2024 - Current

**Objective:**  
To process and analyze large-scale datasets efficiently using **PySpark** on the **Databricks** platform. The aim was to handle big data challenges and create scalable, optimized data pipelines.

**Process:**  
- **Data Import**: Used PySpark's **RDDs (Resilient Distributed Datasets)** and **DataFrames** for importing and processing large datasets across distributed systems.  
- **Data Transformation**: Performed data cleaning, aggregation, and transformation tasks using Spark SQL and DataFrame APIs.  
- **Optimization**: Optimized data processing pipelines for performance, reducing processing time significantly by leveraging partitioning, caching, and parallel processing.  
- **Big Data Processing**: Processed datasets too large for traditional systems, providing real-time insights and analytics for business needs.  
- **Visualization**: Visualized processed data using integrated Databricks notebooks for in-depth analysis.

**Outcome:**  
- Created scalable data processing pipelines that allowed the analysis of massive datasets, providing insights in real-time.  
- Improved data analysis efficiency, enabling faster decision-making for data-driven business strategies.

**Tools Used:**  
- **PySpark**  
- **Databricks**, **Apache Spark**

---

## 🛠 Skills  
<a id="-skills"></a>  
Here is a list of my key technical and soft skills.

### **Technical Skills:**  
- **Programming Languages**: Python, R, SQL  
- **Machine Learning**: Classification, Regression, Clustering, Deep Learning  
- **Big Data & Cloud**: Spark, Hadoop, AWS, GCP  
- **Data Visualization**: Power BI, Tableau, Matplotlib, Seaborn  
- **Statistical Analysis**: Hypothesis Testing, ANOVA, Regression Analysis  

### **Soft Skills:**  
- **Problem Solving**  
- **Communication**  
- **Collaboration**  
- **Adaptability**  
- **Critical Thinking**

---

## ⚙ Tools & Technologies  
<a id="-tools--technologies"></a>  
I am proficient in a range of tools and technologies that help me effectively analyze data and develop insights.

- **Programming Languages**: Python, R, SQL  
- **Libraries/Frameworks**: Pandas, NumPy, Scikit-learn, TensorFlow, PyTorch  
- **Databases**: MySQL, PostgreSQL, MongoDB  
- **Big Data & Cloud**: AWS, Azure, GCP, Hadoop, Spark  
- **Data Visualization**: Power BI, Tableau, Matplotlib, Seaborn  
- **Development Tools**: Jupyter Notebook, VS Code, Git

---

## 💼 Work Experience  
<a id="-work-experience"></a>  
I have gained hands-on experience in various data science roles, where I applied my skills to solve real-world business challenges.

### **Data Visualization Analyst (Part-Time)**  
**Eagle Cars & Tiger Taxis | Oct 2024 - Present | Clitheroe, UK**  
- Created weekly and monthly dashboards to report driver performance.  
- Automated reporting processes, reducing manual reporting time by 50%.  
- Developed visualizations that improved decision-making for stakeholders.

### **Data Scientist (Full-Time)**  
**WebDoc | May 2023 - Dec 2023 | Islamabad, Pakistan**  
- Improved data accuracy by 20% through data cleaning and validation.  
- Created over 15 dynamic visualizations to represent complex datasets.  
- Applied regression and classification models to predict user behavior.

### **Data Insights Analyst (Full-Time)**  
**Zones, IT Solutions | Sep 2021 - May 2023 | Islamabad, Pakistan**  
- Developed data-driven strategies that increased customer retention by 18%.  
- Designed and maintained Power BI dashboards for real-time performance tracking.  
- Collaborated with cross-functional teams to design reports for business decisions.

---

## 🎓 Education  
<a id="-education"></a>  
Here’s my academic background that laid the foundation for my career in data science.

### **M.S. in Data Science** _(Expected May 2025)_  
**University of Salford, UK**  
- Coursework: Machine Learning, Big Data Analytics, NLP, Deep Learning

### **B.S. in Software Engineering** _(Graduated May 2022)_  
**Bahria University, Pakistan**  
- Coursework: AI, Data Mining, Web Development, Database Systems

---

## 🎯 Activities  
<a id="-activities"></a>  
In addition to my professional and academic pursuits, I am actively involved in extracurricular activities.

### **President, Dawah Society - Salford University** _(2024)_  
- Organized weekly social events to foster student engagement and unity.

---

## 🔧 How to Use This Repository  
<a id="-how-to-use-this-repository"></a>  
Clone this repository to explore my projects and codebase:  
```bash
git clone https://github.com/your-username/data-scientist-portfolio.git  
cd data-scientist-portfolio

## 📞 Contact
<a id="-contact"></a>
You can get in touch with me through the following channels:

📧 Email: your.email@example.com
🔗 LinkedIn: Your LinkedIn Profile
🐙 GitHub: Your GitHub Profile
