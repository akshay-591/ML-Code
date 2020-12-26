package com.JML.KMean.algo;


import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import java.util.HashMap;
import java.util.Map;


/**
 * This class Contains Centroids Methods for Finding and calibrating.
 * @author Akshay Kumar
 */
public class Centroids {
    /**
     * This method will Calculate the Distance between the Data Point and The Centroids.
     *
     * @param input            - is the Input Matrix which Contain the Data without label
     * @param initialCentroids - initialCentroids with respect to which distance will be calculated
     * @return Map<String,INDArray> Containing Two IND Arrays 1st is INDArray which will contain the minimum distance.
     * <p>
     * for ex- we will calculate the distance between every data point w.r.t each centroids then
     * we will get three distances for each data point/example and out of those three which will
     * be the minimum we will going to considered that distance.
     * Key for That Array is "minDist"
     * <p>
     * At index 1 in the List Indexes of those minimum distance will going to be Stored.
     * <p>
     * for ex - IF out of three distance minimum is at column/row 2 that means that data point/example
     * belongs to 2nd Centroids.
     * Key for That Array is "minIdx"
     */
    public static Map<String,INDArray> findNearestCentroids(INDArray input, INDArray initialCentroids) {
        INDArray temp = Nd4j.zeros(input.rows(), initialCentroids.rows());


        for (int i = 0; i < initialCentroids.rows(); i++) {
            INDArray distance = input.sub(initialCentroids.getRow(i));
            distance = Transforms.pow(distance, 2); // square the distance
            INDArray squareDistance = Nd4j.sum(distance, 1);
            temp.putColumn(i, squareDistance);
        }

        // find minimum distance for each data point
        INDArray minimumDistance = Nd4j.min(temp, 1);
        INDArray minimumDistanceIndex = Nd4j.argMin(temp, 1);
        Map<String,INDArray> result = new HashMap<>();
        result.put("minDist",minimumDistance);
        result.put("minIdx",minimumDistanceIndex);

        return result;
    }

    /**
     * This method will calibrate/change the centroids.
     *
     * @param input        - input Data points/Examples without label
     * @param indexes      minimum Distance indexes calculates using nearestCentroids() method.
     * @param numCentroids - Number of Centroids.
     * @return Calibrated or new Centroids Containing in INDArray.
     */
    public static INDArray calibrateCentroids(INDArray input, INDArray indexes, int numCentroids) {
        INDArray newCentroids = Nd4j.zeros(numCentroids, input.columns()).castTo(DataType.DOUBLE);

        for (int i = 0; i < numCentroids; i++) {
            INDArray[] ind = Nd4j.where(indexes.eq(i),null, null);
            INDArray dataPoints = input.get(ind[0]);
            newCentroids.putRow(i, dataPoints.mean(0));
        }
        return newCentroids;

    }
}
