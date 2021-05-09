package Regression.Optimizers;

import org.nd4j.linalg.api.buffer.DataType;
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

     private INDArray learnedWeights;


    /**
     * This Method Calculate the Prediction Value and Return the Prediction as an Object of INDArray class.
     *
     *
     * @param X are the Input values in INDArray form This array/Matrix should contain examples in
     *          rows(along the axis 0) and Features in Columns(along the axis 1).
     *
     * @param weights are the Learned or Initial weights.
     *
     * @return Return Value from this Method will going to be an INDArray object.
     */

    public INDArray predict(INDArray X, INDArray weights){

        INDArray prediction;
        int xColumns = X.columns();
        int wRows = weights.rows();
        if (X.dataType() !=weights.dataType()){
            X = X.castTo(DataType.DOUBLE);
            weights = weights.castTo(DataType.DOUBLE);
        }
        if (xColumns==wRows){
            prediction = Nd4j.matmul(X,weights);
        }
        else {
            prediction = Nd4j.matmul(X, weights.transpose());
        }
        return prediction;


    }
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
        INDArray pValues =  predict(x,weights);
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

        for (int i=0;i<=iteration;i++) {
            pValues = predict(x, optimumGrads);

            //calculating error
            error = pValues.sub(y);

            //minimizing cost
            miniCost = Nd4j.matmul(x.transpose(), error);

            gradMinimize = miniCost.mul((double)alpha/nExamples);

            optimumGrads = optimumGrads.sub(gradMinimize);

        }

        return optimumGrads;
    }

    /**
     * This Method Calculates The derivative for Least Square Function.
     *
     * J(W) = 1/m*(X*W - Y)*X
     *
     * @param x INDArray object Which contains the data Samples.
     * @param y INDArray object Which contains the Original Labels
     * @param weights INDArray object which contains the Weights/parameters
     * @return INDArray object which derivative of the function.
     */
    public INDArray jacobLeastSquare(INDArray x,INDArray y,INDArray weights){

        INDArray error;
        INDArray miniCost;
        INDArray pValues;
        int nExamples = x.rows();

        pValues = predict(x, weights);

        //calculating error
        error = pValues.sub(y);

        //minimizing cost
        miniCost = Nd4j.matmul(x.transpose(), error);
        return miniCost.mul((double)1.0/nExamples);
    }

    /**
     * This Method Calculate the gradients using Conjugate gradient Method
     *
     * @param x INDArray object
     * @param y
     * @param weights
     * @param maxIter
     * @return
     */
    public INDArray ConjugateGradient(INDArray x,INDArray y,INDArray weights,int maxIter){



        return null;
    }

}
