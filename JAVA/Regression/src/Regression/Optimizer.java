package Regression;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This Class Contain Methods for Calculating Cost/loss Function and Gradient Descent for Regression Models
 *
 *
 * @author Akshay Kumar on 29-07-2020 at 22:48
 * @package com.ML.Regression
 * @project ML_Linear Regression
 *
 */
public class Optimizer {

    PredictionHypothesis prediction;

    /**
     * This Method calculate the cost or error for Regression Model
     *
     * @param x is the Input values and should be a object of INDArray class.Rows of the input should contain
     *          the example and columns should the features
     * @param y is the original output values and should be a object of INDArray class.
     *          the output should be in rows.
     * @param weights and Learned or initial weights/parameter and should be an object of INDArray class.
     * @return cost value as an object of INDArray class.
     */

    public INDArray calculateCost(INDArray x,INDArray y,INDArray weights){
        int nExamples = x.rows();
        INDArray cost;
        INDArray error;
        INDArray squareError;

        //Calculate Prediction
        prediction = new PredictionHypothesis();
        INDArray pValues =  prediction.Regression(x,weights);
        //calculating error and square error
        error = pValues.sub(y);
       //calculating square error
        squareError = Nd4j.matmul(error.transpose(),error);

        //calculating the cost
        cost = squareError.mul((double)1/(2*nExamples));

        return cost;
    }

    /**
     * This method calculate the optimum weights or minimizes the cost/error for Regression model
     *
     * @param x is the Input values and should be a object of INDArray class.Rows of the input should contain
     *           the example and columns should the features
     * @param y is the original output values and should be a object of INDArray class.
     *          the output should be in rows.
     *@param weights and Learned or initial weights/parameter and should be an object of INDArray class.
     * @param alpha is the step value or learning rate
     * @param iteration are the number iteration want to perform in cost function
     * @return optimized weights also known as Learned weights.
     */

    public INDArray calculateGradient (INDArray x,INDArray y,INDArray weights, double alpha,int iteration){
        INDArray optimumGrads = weights;
        int nExamples = x.rows();
        INDArray error;
        INDArray miniCost;
        INDArray pValues;
        INDArray gradMinimize;
        double[] loop = new double[iteration+1];
        double[] cost = new double[iteration+1];
        INDArray costLoop;

        prediction = new PredictionHypothesis();

        for (int i=0;i<=iteration;i++) {
            pValues = prediction.Regression(x, optimumGrads);

            //calculating error and square error
            error = pValues.sub(y);

            //minimizing cost
            miniCost = Nd4j.matmul(x.transpose(), error);

            gradMinimize = miniCost.mul((double)alpha/nExamples);

            optimumGrads = optimumGrads.sub(gradMinimize);
            loop[i] =i;
            costLoop = this.calculateCost(x,y,optimumGrads);
            cost[i]=costLoop.getDouble();

        }
         /*INDArray c1 = Nd4j.create(loop, new int[]{iteration+1, 1});
         INDArray c2 = Nd4j.create(cost,new int[]{iteration+1,1});
         //System.out.println("c1= "+Arrays.toString(c1.transpose().shape())+"\nc2= "+Arrays.toString(c2.transpose().shape()));
         costLoop = Nd4j.concat(1,c1,c2);
        JMLPlot plotter = new JMLPlot();
        plotter.setChartLabel("Testing");
        plotter.setxAxisLabel("iteration");
        plotter.setyAxisLabel("cost");
        plotter.plotScatter(costLoop);*/

        return optimumGrads;
    }

}
