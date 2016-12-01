import pandas as pd
import matplotlib.pyplot as plt
import tkFileDialog
import numpy as np
import scipy.stats as st
#import seaborn as sns
from argparse import ArgumentParser
import os.path

############################################################
# Classes and functions
############################################################

# Create classes: plot parameters, ItemTestData
class PlotParameters:
    def __init__(self, totalCount, target, x_axis_min,x_axis_max):
        # Optional: sets the total number of tags to reach 100%  
        # If not all tags were read, the total tag count can be input here.      
        self.totalCount = totalCount
        # Sets the target percentage for the application
        self.target = target
        # Sets the x-axis min and max values for transmit power in dBm
        self.x_axis_min = x_axis_min
        self.x_axis_max = x_axis_max
        # Create datafile for plotting y-axis
        data = np.array((1,10,50,90,95,99))
        ydf = pd.DataFrame(data, columns=['Pstr'])
        ydf['P']=ydf['Pstr']/100
        ydf['Z'] = st.norm.ppf(ydf['P'])
        self.ydf = ydf

# Class to import test results and process data
class TestResults(PlotParameters):#PlotParameters):
    def __init__(self,fileName,name,parameters):
        # Inherit attributes from PlotParameters class
        #PlotParameters.__init__(self)
        PlotParameters.__init__(self, parameters.totalCount, parameters.target, parameters.x_axis_min, parameters.x_axis_max)
        self.fileName = fileName
        # Data does NOT need to be pre-sorted by power
        try:
            self.rawdata = pd.read_csv(fileName, sep='\t', names=['Tags Read', 'Read Number', 'Read Number 2', 'Average RSSI', 'Antenna', 'Power'])
        except NameError:
            # If undefined, import failed
            self.importSuccess = False
        else:
            # Import succeeded
            self.importSuccess = True   
        self.name = name
        #self.parameters = parameters
        
    # Check if the import succeeded on init
    #def importFile(self):
     #   pass
    def processData(self):
        # data = processed data
        # Slice dataframe and group by total tags read for a given power level
        df1 = self.rawdata[['Tags Read', 'Power']]
        
        #Populate dataframe with empty data
        i=self.x_axis_min
        j=1
        a=[]
        index=[]
        
        while i<=self.x_axis_max:
            a.append(i)
            index.append(str(j))
            i+=1            
            j+=1
        
        # Create empty dataframe with all power levels to plot in dataset and append to dataframe
        d = {'Power' : pd.Series(a, index=index)}
        
        df2 = pd.DataFrame(d)
        
        df = df1.append(df2,ignore_index=True)
        
        df=df.groupby(['Power']).count()
        # Power column turned in to index -> ce-create column from index
        df['Power'] = df.index
        # Make a new index
        df = df.set_index([range(df.shape[0])])
        self.dft = df
        
        # Create a column for a running total of tags read
        df['Tag Count']=df['Tags Read'].cumsum()
        #df1['Tag Count']=df1['Tags Read'].cumsum()
        
        if (self.totalCount!=""):
            df['Total']=self.totalCount
            #df1['Total']=self.totalCount
        else:
            self.totalCount=df['Tags Read'].sum()
            df['Total']=self.totalCount
            #df1['Total']=self.totalCount
        
        df['Total Percentage']=df['Tag Count']/df['Total']
        #df1['Total Percentage']=df1['Tag Count']/df1['Total']
        
        # Calculate effective Z values based on measured percentage. Default 0% to 0.01% and 100% to 99%
        df['Eff Z']=st.norm.ppf(df['Total Percentage'])
        df[df[['Eff Z']]<-2.326] = -2.326 # <0.01% cumulative % = 0.01 or z = -2.326
        df[df[['Eff Z']]>2.326] = 2.326 # >0.99% cumulative % = 0.99 or z = 2.326
        
        self.df = df
        #df1['Eff Z']=st.norm.ppf(df1['Total Percentage'])
        #df1[df1[['Eff Z']]<-2.326] = -2.326 # <0.01% cumulative % = 0.01 or z = -2.326
        #df1[df1[['Eff Z']]>2.326] = 2.326 # >0.99% cumulative % = 0.99 or z = 2.326
        
        # Slice out "noisy" data, where 0% and 100% of tags are read
        df3 = df
        df3=df3[df3['Total Percentage'] != 0]
        df3=df3[df3['Total Percentage'] != 1]
        
        ### Calc the trendline of dataframe from >0% to <100%
        model = np.polyfit(df3['Power'], df3['Eff Z'], 1)
        self.model = model
        predicted = np.polyval(model,df['Power'])
        self.predicted = predicted
        
        # Calculate standard deviation
        # std_dev = 1 / m
        std_dev = 1 / model[0]  
        self.std_dev = std_dev
        # Calculate mean power level needed per tag
        # mean = x when z(x)=0 or cumulative distribution = 50%, or b/m
        mean = - model[1]/model[0]
        self.mean = mean
        
        # Calculate mean and standard deviation of data points from the model
        df3['Z']=(df3['Power']*1.0 - mean)/std_dev
        df3['Z difference']=df3['Eff Z'] - df3['Z']
        m2=df3['Z difference'].mean()
        self.m2 = m2
        s2 = df3['Z difference'].std()
        self.s2 = s2
    def calculateTarget(self):
        # Calculate reader power needed to reach target cumulative percentage
        targetZ = st.norm.ppf(self.target) * 1.0
        # y = mx + b => x = (y - b)/m = y/m - b/m = Z*sigma + mean
        #targetP = targetZ * std_dev + mean
        self.targetP = (targetZ - self.model[1])/self.model[0]
        # Worse-case scenario to high target percentage
        # x = (y - (b - |s2|))/m
        self.targetP_conf = (targetZ - self.model[1] + abs(self.s2))/self.model[0]
    
#class CreatePlots(PlotParameters):
class CreatePlots(PlotParameters):
    # Input is instance of PlotParameters class
    def __init__(self, title, parameters):
        # Inherit attributes from PlotParameters class
        #PlotParameters.__init__(self)
        PlotParameters.__init__(self, parameters.totalCount, parameters.target, parameters.x_axis_min,parameters.x_axis_max)
        plt.xlabel('Reader Power, Pr (dBm)')
        plt.ylabel('Cumulative Distribution (%)')
        plt.title(title)
        self.title = title
        plt.grid(True)
        # Set x-axis to min and max power values
        plt.axis([self.x_axis_min, self.x_axis_max, -2.326, 2.326])
        # Use lookup table for Y axis values
        plt.yticks(self.ydf['Z'], self.ydf['Pstr'], rotation='horizontal')
    # Input is instance of TestResults class
    # Add the results from TestResults class and create new plot.
    def addData(self, results, plot_color,confidenceInterval, model):
        #self.fileName = fileName
        self.name = results.name
        self.confidenceInterval = confidenceInterval
        # If true, plot std_dev and mean on plot        
        self.model = model
                # Plot normalized results
        plt.plot(results.df['Power'], results.df['Eff Z'],"o-",color=plot_color)
        #plt.text(18, 2, r'$\mu={0:.1f},\ \sigma={1:.2f}$dB'.format(mean, std_dev))    
        
        # Plot the trendline
        plt.plot(results.df['Power'],results.predicted,color=plot_color,lw=3)
        # Plot confidence interval of 68%, or +/- 1 std_dev of measured values from the trendline
        if self.confidenceInterval == True:
            plt.plot(results.df['Power'],results.predicted+results.s2,'--',color=plot_color)
            plt.plot(results.df['Power'],results.predicted-results.s2,'--',color=plot_color)
        if self.model == True:
            plt.text(18, 2, r'$\mu={0:.1f},\ \sigma={1:.2f}$dB'.format(results.mean, results.std_dev))
    def plotData(self):
        
        plt.show()
        # Plot all the things
        
# Argument parser
def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg # return the file name
        
        
def main():
    ############################################
    # Main
    ############################################
    param = PlotParameters("",.01,12,35)
    results1 = TestResults("E62StackedJeans.csv", "AutoTune Enabled", param)
    # If the file was imported successfully, process the data
    if results1.importSuccess == True:
        results1.processData()
        results1.calculateTarget()
    else:
        print "The import of the file failed. Is the path correct? " + results1.name
    results2 = TestResults("E62StackedJeans.csv", "AutoTune Disabled", param)
    # If the file was imported successfully, process the data
    if results2.importSuccess == True:
        results2.processData()
        results2.calculateTarget()
    else:
        print "The import of the file failed. Is the path correct? " + results2.name
    #    print "The import of the file " + results2.name " failed. Is the path correct?"
        
    plots = CreatePlots("Stacked Jeans - E62",param)
    plots.addData(results1, "red", True, True)
    #plots.addData(results2, "r", True, False)
    
    print "This application would hit the target percentage {0:.0f}% at the\
    reader Tx power of {1:.1f} dBm".format(param.target*100, results1.targetP)
    print "sigma (1): {0:.1f}".format(results1.std_dev)
    print "With high confidence, this application would hit the target \
    percentage at the reader Tx power of {0:.1f} dBm".format(results1.targetP_conf)
    
    print "For the second set of results, the needed reader TX power is \
    {0:.1f} dBm and {1:.1f} dBm with high confidence."\
    .format(results2.targetP,results2.targetP_conf)
    
    plots.plotData()
    
    ############################################
    # End Main
    ############################################


if __name__ == "__main__":
    main()
    
    

## Import text via comman prompt
#parser = ArgumentParser(description="Input CSV file")
#parser.add_argument("-f", dest="filename", required=False,
#                    help="Input a CSV file of the ItemTest MarginTest results", metavar="FILE",
#                    type=lambda x: is_valid_file(parser, x))
##parser.add_argument("-r", "--range", dest="range", required=False,
##                    help="Set the x axis range for transmit power", action="store_const")
#args = parser.parse_args()
#
#if str(args.filename) == "None":
#    # If no parameters given or parameters are incorrect, open file manually
#    DATA_FILE_NAME = tkFileDialog.askopenfilename(filetypes=[("MarginTest CSV files", "*.csv")])
#else:
#    # Set the filename to the value input by user
#    DATA_FILE_NAME = args.filename
#




