package com.JML.KMean.algo;
import org.nd4j.linalg.api.ndarray.INDArray;
import java.util.HashMap;
import java.util.Map;

/**
 * This Class Contains Methods for initializing KMeans Algorithm.
 *
 * @author Akshay Kumar
 */
public class KMeans {

    /**
     * This Methods will initialize the KMeans Algorithm and The best Centroids for The Data Points.
     * @param Data Unlabelled Data.
     * @param centroids initial centroids
     * @param maxIter maximum Number of Iteration which user wants to perform.
     * @return Map<String, INDArray> Containing Two INDArray The Final Centroids (Key = "Centroids")
     *                               and Indexes (Key = "minIdx") of Centroids for The Data Points belongs to.
     */
    public static Map<String,INDArray> initKMeans(INDArray Data, INDArray centroids, int maxIter, int numCentroids){
        Map<String,INDArray> result = new HashMap<>();
        for (int i =0; i<=maxIter;i++){
            // find nearest Centroids
            result = Centroids.findNearestCentroids(Data,centroids);
            centroids = Centroids.calibrateCentroids(Data,result.get("minIdx"),numCentroids);

        }

        result = Centroids.findNearestCentroids(Data,centroids);
        result.put("Centroids", centroids);

        return result;
    }
}
