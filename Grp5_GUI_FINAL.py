# ___________________________________________
# Data Mining Group 5 Final Project
# Jyoti Sharma, Tanvi Hindwan, and Jessa Henderson
# Features impacting price in Florence Airbnbs
# ___________________________________________

# Import Proper Packages

import sys
from PyQt5.QtWidgets import QMainWindow, QAction, QMenu, QApplication, QGroupBox, QLineEdit, QPushButton, QPlainTextEdit
from PyQt5.QtWidgets import QCheckBox
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import  QWidget,QLabel, QVBoxLayout, QHBoxLayout, QGridLayout, QSizePolicy

import pandas as pd
import seaborn as sns
import numpy as np
from scipy.stats import norm
import scipy
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from numpy.polynomial.polynomial import polyfit
from datetime import datetime, timedelta
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------
# Deafault font size for all the windows
# --------------------------------
font_size_window = 'font-size:16px'

#Setup Classes for Each Drop Down Item
class PriceDistribution(QMainWindow):
    send_fig = pyqtSignal(str)

    #--------------------------------------------------------
    # This class if for the price distribution as shown via box plot
    #--------------------------------------------------------

    def __init__(self):
        #--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #--------------------------------------------------------
        super(PriceDistribution, self).__init__()

        self.left = 200
        self.top = 200
        self.Title = 'Histogram for Price'
        self.width = 500
        self.height = 500
        self.initUi()

    def initUi(self):
        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)

        self.setGeometry(self.left, self.top, self.width, self.height)

        self.m = PlotCanvas(self, width=5, height=4)
        self.m.move(0,30)

class PlotCanvas(FigureCanvas):
    #::----------------------------------------------------------
    # creates a figure on the canvas
    # later on this element will be used to draw plots/graphs
    # this is used by multiple classes
    #::----------------------------------------------------------
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self):
        self.ax = self.figure.add_subplot(111)

#The next class is for Feature Selection

class RandomForest(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        #--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #--------------------------------------------------------
        super(RandomForest, self).__init__()
        self.Title = 'Feature Selection Using Random Forest'
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the elements to create a dashboard
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.main_widget = QWidget(self)
        self.layout = QHBoxLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Feature Selection Demo')
        self.groupBox1Layout = QHBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.btnExecute = QPushButton("Execute RF")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1Layout.addWidget(self.btnExecute)

        self.layout.addWidget(self.groupBox1)

        #::-------------------------------------------
        # Graphic 1: Feature Analysis
        #::-------------------------------------------

        self.fig1 = Figure()
        self.ax1 = self.fig1.add_subplot(111)
        self.axes1 = [self.ax1]
        self.canvas1 = FigureCanvas(self.fig1)

        self.canvas1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas1.updateGeometry()

        self.groupBoxG1 = QGroupBox('Feature Analysis - Top 25')
        self.groupBoxG1Layout = QHBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)
        self.groupBoxG1Layout.addWidget(self.canvas1)

        #::-------------------------------------------
        # Graphic 2: Feature Analysis Verification
        #::-------------------------------------------

        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)

        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas2.updateGeometry()

        self.groupBoxG2 = QGroupBox('Feature Analysis Verification with RF')
        self.groupBoxG2Layout = QHBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)
        self.groupBoxG2Layout.addWidget(self.canvas2)

        #::-------------------------------------------------
        # End of graphs
        #::-------------------------------------------------

        self.layout.addWidget(self.groupBox1)
        self.layout.addWidget(self.groupBoxG1)
        self.layout.addWidget(self.groupBoxG2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):

        # Extract features and labels
        X_dt = AirbnbFeatures.drop('price', axis=1)
        y_dt = AirbnbFeatures['price']

        # perform training with random forest with all columns
        # specify random forest regressor
        clf = RandomForestRegressor(n_estimators=100)

        # Training and Testing Sets
        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=0.2, random_state=0)

        # perform training
        clf.fit(X_train, y_train)

        #::------------------------------------
        ##  Graph1 : Feature Analysis
        #::------------------------------------

        # plot feature importances
        # get feature importances
        importances = clf.feature_importances_

        # convert the importances into one-dimensional 1-d array with corresponding df column names as axis labels
        f_importances = pd.Series(importances, AirbnbFeatures.drop(columns='price').columns)

        # sort the array in descending order of the importances
        f_importances.sort_values(ascending=False, inplace=True)

        self.ax1.barh(f_importances.index[0:25], f_importances.values[0:25])
        self.ax1.set_aspect('auto')

        # show the plot
        self.fig1.tight_layout()
        self.fig1.canvas.draw_idle()

        #::------------------------------------
        ##  Graph2 : Feature Analysis Verification with Random Forest
        #::------------------------------------

        # Create a selector object that will use the random forest regressor to identify features
        select_features = SelectFromModel(RandomForestRegressor(n_estimators=100))  # estimators are the number of trees
        select_features.fit(X_train, y_train)

        # In order to check which features among all important we can use the method get_support()
        select_features.get_support()

        # This method will output an array of boolean values.
        # True for the features whose importance is greater than the mean importance and False for the rest.

        # create list and count features
        selected_feature = X_train.columns[(select_features.get_support())]
        nlarge = f_importances.nlargest(22)

        self.ax2.barh(nlarge.index, nlarge.values)
        self.ax2.set_aspect('auto')

        # show the plot
        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

class RFperformance(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        #--------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        #--------------------------------------------------------
        super(RFperformance, self).__init__()
        self.Title = 'Feature Selection Performance'
        self.initUi()

    def initUi(self):
        #::-----------------------------------------------------------------
        #  Create the canvas and all the elements to create a dashboard
        #::-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.main_widget = QWidget(self)
        self.layout = QVBoxLayout(self.main_widget)

        self.groupBox1 = QGroupBox('Feature Selection Performance')
        self.groupBox1Layout = QVBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.btnExecute = QPushButton("Execute Performance Check")
        self.btnExecute.clicked.connect(self.update)

        self.checkbox1 = QCheckBox('Show Regression Line', self)
        self.checkbox1.stateChanged.connect(self.update)

        self.label1 = QLabel('Accuracy score:')
        self.label2 = QLabel('Mean Squared Error:')
        self.label3 = QLabel('Root Mean Squared Error:')

        self.layout.addWidget(self.groupBox1)
        self.groupBox1Layout.addWidget(self.btnExecute)
        self.layout.addWidget(self.checkbox1)
        self.layout.addWidget(self.label1)
        self.layout.addWidget(self.label2)
        self.layout.addWidget(self.label3)

        #::-------------------------------------------
        # Graphic 1: Feature Analysis
        #::-------------------------------------------

        self.fig1 = Figure()
        self.ax1 = self.fig1.add_subplot(111)
        self.axes1 = [self.ax1]
        self.canvas1 = FigureCanvas(self.fig1)

        self.canvas1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas1.updateGeometry()

        self.groupBoxG1 = QGroupBox('Residual Plot')
        self.groupBoxG1Layout = QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)
        self.groupBoxG1Layout.addWidget(self.canvas1)
        self.layout.addWidget(self.groupBoxG1)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):

        # Extract features and labels
        X_dt = AirbnbFeatures.drop('price', axis=1)
        y_dt = AirbnbFeatures['price']

        # perform training with random forest with all columns
        # specify random forest Regressor
        clf = RandomForestRegressor(n_estimators=100)

        # Training and Testing Sets
        X_train, X_test, y_train, y_test = train_test_split(X_dt, y_dt, test_size=0.2, random_state=0)

        # perform training
        clf.fit(X_train, y_train)

        #This runs the residual plot again based on if the regression line is chosen
        self.ax1.clear()

        from sklearn.metrics import mean_squared_error, accuracy_score
        y_pred = clf.predict(X_train)

        # Use the model to predict values
        y_pred = clf.predict(X_test)

        # Plot of model's residuals:
        self.ax1.plot(y_test, y_pred, 'bo')
        self.ax1.set_aspect('auto')

        # show the plot
        self.fig1.tight_layout()
        self.fig1.canvas.draw_idle()

        if self.checkbox1.isChecked():
            b, m = polyfit(y_test, y_pred, 1)

            self.ax1.plot(y_test, b + m * y_test, '-', color="orange")

        vtitle = "Residual Plot "
        self.ax1.set_title(vtitle)
        self.ax1.set_xlabel("Price")
        self.ax1.grid(True)

        self.fig1.tight_layout()
        self.fig1.canvas.draw_idle()

        self.label1.setText("Accuracy score fn: %.3f" % clf.score(X_test, y_test))
        self.label2.setText('Mean Squared Error : %0.3f' % mean_squared_error(y_test,y_pred))
        self.label3.setText("Root Mean Squared Error : %0.3f" % (mean_squared_error(y_test,y_pred))**0.5)


#The next class is for model analysis
class LinearRegression(QMainWindow):
    send_fig = pyqtSignal(str)

    def __init__(self):
        # --------------------------------------------------------
        # Initialize the values of the class
        # Here the class inherits all the attributes and methods from the QMainWindow
        # --------------------------------------------------------
        super(LinearRegression, self).__init__()
        self.Title = 'Linear Regression: Features Predicting Price'
        self.initUi()

    def initUi(self):
        #-----------------------------------------------------------------
        #  Create the canvas and all the elements to create a dashboard
        #-----------------------------------------------------------------

        self.setWindowTitle(self.Title)
        self.setStyleSheet(font_size_window)
        self.main_widget = QWidget(self)
        self.layout = QGridLayout(self.main_widget)

        self.btnExecute = QPushButton("Execute LR")
        self.btnExecute.clicked.connect(self.update)

        self.groupBox1 = QGroupBox('Linear Regression V1')
        self.groupBox1Layout = QVBoxLayout()
        self.groupBox1.setLayout(self.groupBox1Layout)

        self.label1 = QLabel('Coefficients:')
        self.label2 = QLabel('Mean Squared Error:')
        self.label3 = QLabel('R2 score:')

        self.groupBox2 = QGroupBox('Linear Regression V2')
        self.groupBox2Layout = QVBoxLayout()
        self.groupBox2.setLayout(self.groupBox2Layout)

        self.label5 = QLabel('Coefficients:')
        self.label6 = QLabel('Mean Squared Error:')
        self.label7 = QLabel('R2 score:')

        self.layout.addWidget(self.groupBox1, 0, 0)
        self.layout.addWidget(self.groupBox2, 0, 1)
        self.groupBox1Layout.addWidget(self.btnExecute)
        #self.layout.addWidget(self.label1, 1, 0)
        self.layout.addWidget(self.label2, 1, 0)
        self.layout.addWidget(self.label3, 2, 0)
        #self.layout.addWidget(self.label5, 1, 1)
        self.layout.addWidget(self.label6, 1, 1)
        self.layout.addWidget(self.label7, 2, 1)

        #-------------------------------------------
        # Graphic 1: Linear Regression V1
        #-------------------------------------------

        self.fig1 = Figure()
        self.ax1 = self.fig1.add_subplot(111)
        self.axes1 = [self.ax1]
        self.canvas1 = FigureCanvas(self.fig1)

        self.canvas1.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas1.updateGeometry()

        self.groupBoxG1 = QGroupBox('Linear Regression V1')
        self.groupBoxG1Layout = QVBoxLayout()
        self.groupBoxG1.setLayout(self.groupBoxG1Layout)
        self.groupBoxG1Layout.addWidget(self.canvas1)
        self.layout.addWidget(self.groupBoxG1)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

        #-------------------------------------------
        # Graphic 2: Linear Regression V2
        #-------------------------------------------

        self.fig2 = Figure()
        self.ax2 = self.fig2.add_subplot(111)
        self.axes2 = [self.ax2]
        self.canvas2 = FigureCanvas(self.fig2)

        self.canvas2.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.canvas2.updateGeometry()

        self.groupBoxG2 = QGroupBox('Linear Regression V2')
        self.groupBoxG2Layout = QVBoxLayout()
        self.groupBoxG2.setLayout(self.groupBoxG2Layout)
        self.groupBoxG2Layout.addWidget(self.canvas2)
        self.layout.addWidget(self.groupBoxG2)

        self.setCentralWidget(self.main_widget)
        self.resize(1100, 700)
        self.show()

    def update(self):

        # # Linear Regression using features from Correlation Matrix
        U = coormatrix_features.drop('price', axis=1)
        V = coormatrix_features['price']

        # split the dataset into train and test
        U_train, U_test, V_train, V_test = train_test_split(U, V, test_size=0.2, random_state=10)

        # Standardize the features and target / # normalizing the features target
        ss = StandardScaler()
        U_train = ss.fit_transform(U_train)
        U_test = ss.transform(U_test)  # borrowing parameters from train
        U_train.shape, U_test.shape
        V_train = ss.fit_transform(V_train.values.reshape(-1, 1))
        V_test = ss.transform(V_test.values.reshape(-1, 1))

        regr = linear_model.LinearRegression()
        regr.fit(U_train, V_train)
        airbnb_V_pred = regr.predict(U_test)

        vtitle = "V1 Prices vs Predicted Prices: $Y_i$ vs $\hat{Y}_i$"
        self.ax1.plot(V_test, airbnb_V_pred, 'bo')
        self.ax1.set_title(vtitle)
        self.ax1.set_xlabel("Prices: $Y_i$")
        self.ax1.set_ylabel("Predicted prices: $\hat{Y}_i$")
        self.ax1.grid(True)

        self.fig1.tight_layout()
        self.fig1.canvas.draw_idle()

        #self.label1.setText('V1 Coefficients: \n' % regr.coef_)
        self.label2.setText("V1 Mean squared error: %.2f" % mean_squared_error(V_test, airbnb_V_pred))
        self.label3.setText('V1 R2 score: %.2f' % r2_score(V_test, airbnb_V_pred))

        #LR - V2
        X = FeaturesFINAL.drop('price', axis=1)
        y = FeaturesFINAL['price']

        # split the dataset into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

        # Standardize the features and target / # normalizing the features target
        ss = StandardScaler()
        X_train = ss.fit_transform(X_train)
        X_test = ss.transform(X_test)  # borrowing parameters from train

        y_train = ss.fit_transform(y_train.values.reshape(-1, 1))
        y_test = ss.transform(y_test.values.reshape(-1, 1))

        regr = linear_model.LinearRegression()
        regr.fit(X_train, y_train)
        airbnb_y_pred = regr.predict(X_test)

        self.ax2.plot(y_test, airbnb_y_pred, 'bo')
        self.ax2.set_title("V2 Prices vs Predicted Prices: $Y_i$ vs $\hat{Y}_i$")
        self.ax2.set_xlabel("Prices: $Y_i$")
        self.ax2.set_ylabel("Predicted prices: $\hat{Y}_i$")
        self.ax2.grid(True)

        self.fig2.tight_layout()
        self.fig2.canvas.draw_idle()

        #self.label5.setText('V2 Coefficients: \n' % regr.coef_)
        self.label6.setText("V2 Mean squared error: %.2f" % mean_squared_error(y_test, airbnb_y_pred))
        self.label7.setText('V2 R2 score: %.2f' % r2_score(y_test, airbnb_y_pred))


# Setup Main Application
class MainWIN(QMainWindow):

    def __init__(self):
        super().__init__()
        self.left = 100
        self.top = 100
        self.width = 500
        self.height = 300
        self.Title = 'Airbnb Prices in Florence, Italy'
        self.setStyleSheet("QWidget {background-image: url(airbnb_logo.png); background-repeat: no-repeat}")
        self.initUI()

    def initUI(self):
        # ::-------------------------------------------------
        # Creates the menu and the items
        # ::-------------------------------------------------
        self.setWindowTitle(self.Title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        self.statusBar()

        # ::-----------------------------
        # Main Menu Creation
        # ::-----------------------------

        mainMenu = self.menuBar()
        mainMenu.setNativeMenuBar(False)
        mainMenu.setStyleSheet('background-color: #FF585D')
        fileMenu = mainMenu.addMenu('File')
        EDAMenu = mainMenu.addMenu('Exploratory Analysis')
        FeatureMenu = mainMenu.addMenu('Feature Selection')
        ModelMenu = mainMenu.addMenu('Model Analysis')

        # ::--------------------------------------
        # Exit action
        # ::--------------------------------------

        exitButton = QAction('&Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.setStatusTip('Exit application')
        exitButton.triggered.connect(self.close)

        fileMenu.addAction(exitButton)

        self.show()

        # ----------------------------------------
        # EDA Buttons
        # Creates the EDA Analysis Menu
        # Price Distribution: Shows the distribution of rental price
        #::----------------------------------------

        EDA1Button = QAction('Price Distribution', self)
        EDA1Button.setStatusTip('Boxplot for Price')
        EDA1Button.triggered.connect(self.EDA1)
        EDAMenu.addAction(EDA1Button)

        # ----------------------------------------
        # Feature Selection Button
        # Creates the Feature Selection Drop Down Menu
        # Random Forest: Random Forest was used to determine the best features for the final model.
        #::----------------------------------------

        FSButton = QAction('Random Forest', self)
        FSButton.setStatusTip('Random Forest for Feature Selection')
        FSButton.triggered.connect(self.FS)
        FeatureMenu.addAction(FSButton)

        #FS2Button = QAction('Random Forest Performance', self)
        #FS2Button.setStatusTip('Random Forest Performance Analysis')
        #FS2Button.triggered.connect(self.FS2)
        #FeatureMenu.addAction(FS2Button)

        # ----------------------------------------
        # Linear Regression Button
        # Creates the Model Analysis Drop Down Menu
        # Linear Regression: Linear Regression was used to investigate how Airbnb features contribute to price
        #::----------------------------------------

        LRButton = QAction('Linear Regression', self)
        LRButton.setStatusTip('Model Analysis with Linear Regression')
        LRButton.triggered.connect(self.LR)
        ModelMenu.addAction(LRButton)

        #:: Creates an empty list of dialogs to keep track of
        #:: all the iterations
        self.dialogs = list()
        self.show()

    # ----------------------------------------
    # EDA Functions
    # Creates the actions for the EDA Analysis Menu
    # EDA1: Amenity Counts
    #::----------------------------------------

    def EDA1(self):
        #::---------------------------------------------------------
        # This function creates an instance of PriceDistribution class
        # This class creates a boxplot that shows price distribution of Florence Airbnbs
        #::---------------------------------------------------------
        dialog = PriceDistribution()
        dialog.m.plot()
        dialog.m.ax.hist(u, bins=25, color='green', alpha=0.5)
        dialog.m.ax.set_title('Price Distribution for Outliers')
        dialog.m.ax.set_xlabel('Price of Airbnbs')
        dialog.m.ax.set_ylabel('Count')
        dialog.m.ax.grid(True)
        dialog.m.draw()
        self.dialogs.append(dialog)
        dialog.show()

    # ----------------------------------------
    # Feature Selection Function
    # Creates the actions for the Feature Selection Menu
    # FS: Feature Selection was conducted using the random forest technique
    #::----------------------------------------

    def FS(self):
        #----------------------------------------------------------
        # This function creates an instance of the RandomForest class
        #----------------------------------------------------------
        dialog = RandomForest()
        self.dialogs.append(dialog)
        dialog.show()

    def FS2(self):
        #----------------------------------------------------------
        # This function creates an instance of the RFperformance class
        #----------------------------------------------------------
        dialog = RFperformance()
        self.dialogs.append(dialog)
        dialog.show()

    # ----------------------------------------
    # Linear Regression Function
    # Creates the actions for the Model Analysis Menu
    # LR: Linear Regression was used as the final model for analysis
    #----------------------------------------

    def LR(self):
        #----------------------------------------------------------
        # This function creates an instance of the LinearRegression class
        #----------------------------------------------------------
        dialog = LinearRegression()
        self.dialogs.append(dialog)
        dialog.show()

#------------------------
# Application starts here
#------------------------
def main():
    app = QApplication(sys.argv)
    app.setStyle('Breeze')
    mn = MainWIN()
    sys.exit(app.exec_())

#------------------------
# Global variables are below
#------------------------

def data_airbnb():
    #--------------------------------------------------
    # Pulls in data and relevant variables and features for the entire GUI.
    #--------------------------------------------------
    global Florencebnb
    global AirbnbFeatures
    global FlorenceFINAL
    global FeaturesFINAL
    global u
    global labels
    global Top10_amenities
    global coormatrix_features
    Florencebnb = pd.read_csv('airbnb_price.csv')
    AirbnbFeatures = pd.read_csv('airbnb_features.csv')
    FlorenceFINAL = pd.read_csv('airbnb_cleaned.csv')
    FeaturesFINAL = pd.read_csv('Regression_features.csv')
    u = Florencebnb.loc[:, 'price']
    coormatrix_features = FeaturesFINAL.loc[:, ('accommodates', 'bathrooms', 'bedrooms', 'security_deposit',
                                        'cleaning_fee', 'guests_included', 'availability_365', 'property_type_Hotel',
                                        'price')]

if __name__ == '__main__':
    data_airbnb()
    main()
