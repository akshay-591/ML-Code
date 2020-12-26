package Regression;


import com.JavaMLVisualizer.UI.JML2DPlot;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.IOException;
import java.util.ArrayList;


/**
 * Testing Linear Regression model Model Using Java
 *
 * @author Akshay Kumar on 29-07-2020 at 19:15
 * @package com.ML.Regression
 * @project ML_Linear Regression
 */

public class LinearRegression {

    public static void main(String[] arg) {
        INDArray loadData;
        INDArray x;
        INDArray xIncludeOnes;
        INDArray y;
        int clm;
        Optimizer optimizer;
        INDArray cost;
        INDArray optimumGrads;
        try {
            // loading data
            loadData = Nd4j.readNumpy("ex1data1.txt", ",");
            clm = loadData.columns();

            /*
             store the data in x and y.Using NDArrayIndex class.
             First parameters here represent number of rows and second represent columns

             Number which passed as second parameter is not include.
             for example if we pass the parameter as (0,3) that means.we extracting only column or row from 0,1,2 not 3.

             all() means all the columns or rows.
             */

            x = loadData.get(NDArrayIndex.all(), NDArrayIndex.interval(0, clm - 1));
            y = loadData.get(NDArrayIndex.all(), NDArrayIndex.interval(clm - 1, clm));
            double xMin = Math.round((Double) x.minNumber());
            double xMax = Math.round((Double) x.maxNumber())+1;

            //Plotting Data
            JML2DPlot plotter = new JML2DPlot();
            plotter.setChartLabel("Regression");
            plotter.setxAxisLabel("Profit");
            plotter.setyAxisLabel("Population");
            plotter.setLegend(true);
            plotter.setLegendTitle("Original");
            plotter.setXAxisRange(xMin,xMax);
            plotter.createXYChart(loadData.transpose(),"+");
            plotter.show();

            // initialize theta
             INDArray weights = Nd4j.zeros(x.columns()+1,1);


             //add ones column to input matrix.
              xIncludeOnes= Nd4j.concat(1,Nd4j.ones(x.rows(),1),x);

             // Calculate cost
             optimizer =new Optimizer();

             cost = optimizer.calculateCost(xIncludeOnes,y,weights);
             System.out.println("Initial cost is  = "+cost);

            // Calculate Gradient
            optimumGrads =optimizer.calculateGradient(xIncludeOnes,y,weights,0.01,1500);
            System.out.println("Optimum weights are = "+optimumGrads);


           // calculate new prediction
            INDArray pre= new PredictionHypothesis().Regression(xIncludeOnes,optimumGrads);
            double[][] dt = loadData.transpose().toDoubleMatrix();
            double[][] dt2 = {x.toDoubleVector(),pre.toDoubleVector()};
            //Plotting data
            ArrayList<double[][]> list = new ArrayList<>();

            // add original dataset
            list.add(dt);
            // add prediction
            list.add(dt2);

            plotter.setLegend(true);
            plotter.setMultiLegendsTitles("original","Prediction");

            plotter.createMultiDataset(list,"+","/");
            plotter.show();





        } catch (IOException ref) {
            ref.printStackTrace();
            System.out.println("error =" + ref);
        }




    }

}
