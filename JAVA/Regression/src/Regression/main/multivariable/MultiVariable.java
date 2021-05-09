package Regression.main.multivariable;

import Regression.Normalization.Scale;
import Regression.Optimizers.Optimizer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.IOException;

/**
 * @author Akshay Kumar 07/05/21
 * @package Regression
 **/
public class MultiVariable {

    public static void main(String[] args) {
        INDArray dataAll;
        INDArray predictors;
        INDArray target;
        INDArray initialWeights;
        int rows;
        int clm;

        String filePath = System.getProperty("user.dir")+"/Data/ex1data2.txt";

        try {
            dataAll = Nd4j.readNumpy(filePath, ",");

            rows = dataAll.rows();
            clm = dataAll.columns();

            predictors = dataAll.get(NDArrayIndex.all(),NDArrayIndex.interval(0, clm - 1));

            // normalized data
            Scale sc = new Scale();
            INDArray normalizedData = sc.minMaxScaler(predictors);

            target = dataAll.get(NDArrayIndex.all(),NDArrayIndex.interval(clm - 1, clm));

            System.out.println("Top 5 values are = \n"+predictors.get(NDArrayIndex.interval(0,5)));

            // includes ones column in predictors

            INDArray prdictorsOnes = Nd4j.concat(1,Nd4j.ones(rows,1),normalizedData);

            initialWeights = Nd4j.zeros(prdictorsOnes.columns(),1);

            // calculate initial cost

            Optimizer opt = new Optimizer();

            INDArray cost = opt.calculateCost(prdictorsOnes,target,initialWeights);
            System.out.println("Initial cost is = "+cost);

            // optimize using Gradient Descent

            INDArray learnedWeights = opt.calculateGradient(prdictorsOnes,target,initialWeights,0.01,1500);

            System.out.println("Optimized weights are = "+learnedWeights);

            // prediction

            System.out.println("Original Values for {2400,3} is  = "+target.getRows(2));
            System.out.println("Predicted Values for {2400,3} is  = "+opt.predict(prdictorsOnes.getRows(2),learnedWeights));


        }
        catch (IOException e){
            e.printStackTrace();
        }
    }
}
