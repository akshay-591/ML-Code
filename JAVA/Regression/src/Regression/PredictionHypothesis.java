package Regression;


import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * This Class Contain Prediction Hypothesis/Equation
 *
 *
 * @author Akshay Kumar on 29-07-2020 at 22:48
 * @package com.ML.Regression
 * @project ML_Linear Regression
 */
public class PredictionHypothesis {

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

    public INDArray Regression(INDArray X,INDArray weights){

        INDArray prediction;
        int xColumns = X.columns();
        int wRows = weights.rows();

        if (xColumns==wRows){
           prediction = Nd4j.matmul(X,weights);
        }
        else {
            prediction = Nd4j.matmul(X, weights.transpose());
        }
        return prediction;


    }

}
