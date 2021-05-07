package Regression.Normalization;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

/**
 * @author Akshay Kumar 07/05/21 2:37 PM
 * @project Regression
 **/
public class Scale {

    private INDArray mean;
    private INDArray std;
    private INDArray variance;
    private INDArray arrayMin;
    private INDArray arrayMax;

    public INDArray getMean() {
        return mean;
    }

    public INDArray getStd() {
        return std;
    }

    public INDArray getVariance() {
        return variance;
    }

    public INDArray getArrayMin() {
        return arrayMin;
    }

    public INDArray getArrayMax() {
        return arrayMax;
    }

    /**
     * Normalization is a scaling technique in which values are shifted and rescaled so that they end up ranging
     * between 0 and 1.
     * This Method perform normalization using mini-max technique.
     *
     * Formula which it used is __
     * X' = X-Xmin / Xmax-Xmin.
     *
     * If data is multi dimension then Normalization is performed on every column.
     *
     * @param array INDArray object containing Examples row wise.
     * @return INDArray object containing Normalized Data.
     */
    public INDArray minMaxScaler(INDArray array){
        this.arrayMin = Nd4j.min(array,0); // get minimum of every column

        this.arrayMax = Nd4j.max(array,0); // get maximum of every column

        // Perform Normalization
        return array.sub(arrayMin).div(arrayMax.sub(arrayMin));

    }

    /**
     * Normalization is a scaling technique in which values are shifted and rescaled so that they end up ranging
     * between 0 and 1.
     * This Method perform normalization using mini-max technique.
     *
     * Formula which it used is __
     * X' = X-Xmin / Xmax-Xmin.
     *
     * If data is multi dimension then Normalization is performed on every column.
     *
     * @param array INDArray object containing Examples row wise.
     * @param Min minimum of the array
     * @param Max maximum of the array
     * @return INDArray object containing Normalized Data.
     */
    public INDArray minMaxScaler(INDArray array,INDArray Min,INDArray Max){
        return array.sub(Min).div(arrayMax.sub(Max));
    }

    /**
     * Standardization is another scaling technique where the values are centered around the mean with a unit
     * standard deviation. This means that the mean of the attribute becomes zero and the resultant distribution
     * has a unit standard deviation.
     *
     * Here’s the formula for standardization:
     * X' = X - mu / sigma
     *
     * Feature scaling: Mu is the mean of the feature values and Sigma is the standard deviation
     * of the feature values.
     *
     * @param array INDArray object containing Examples row wise.
     * @return INDArray Object which contains Scaled Values.
     *
     */

    public INDArray standardScaler(INDArray array){
        mean = Nd4j.mean(array,0); // find mean

        INDArray zeroMean = array.sub(mean); // subtract mean from every feature

        std = Nd4j.std(zeroMean,0); // find standard deviation

        return zeroMean.div(std);
    }
}

