# EECS 731 - Project 2 (Classification)
### Author: Jace Kline

## Project Objectives
1. Set up a data science project structure in a new git repository in your GitHub account
2. Download the Shakespeare plays dataset from https://www.kaggle.com/kingburrito666/shakespeare-plays
3. Load the data set into panda data frames
4. Formulate one or two ideas on how feature engineering would help the data set to establish additional value using exploratory data analysis
5. Build one or more classification models to determine the player using the other columns as features
6. Document your process and results
7. Commit your notebook, source code, visualizations and other supporting files to the git repository in GitHub

## Project Summary
In this project, we carried out the following tasks:
1. Downloaded and loaded the data into a Jupyter notebook
2. Cleaned and prepared the data by...
    * deleting NaN values
    * mapping all strings to lowercase
    * converting object fields to numerical fields
3. Data exploration and understanding
4. Model ideas and exploration
    * Text-based approach using vectorization methods
    * Numerical approach using decision tree methods
5. Results and conclusion

We managed to achieve a 75% model success rate on our test data using the numercial features exclusively and deploying a decision tree model. These two decisions in conjunction allowed us to manage the large amount of data in the dataset to generate a reasonably accurate model given our computational and memory constraints. Future work shall include integration of textual analysis into the current model to produce a more accurate model. Please see the project report [here](asldfj).
