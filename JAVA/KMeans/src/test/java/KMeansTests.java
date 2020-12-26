import static org.junit.Assert.*;

import java.io.IOException;

import java.util.Map;

import com.JML.KMean.algo.KMeans;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import com.JML.KMean.algo.Centroids;


public class KMeansTests {
	private final int numCentroids = 3;
	private final INDArray initial_centroids = Nd4j.create(new double[][] { { 3, 3 },
		                                                              { 6, 2 }, 
		                                                              { 8, 5 } }); // initialize the Testing Centroids
	public static INDArray Data;
	private INDArray minimumDistanceIndex;

	@BeforeClass
	public static void setUpBeforeClass() throws Exception {
		System.out.println("BeforeClass Called");
		// going to Load the Data once Here

		try {
			Data = Nd4j.readNumpy("Data/ex7data2.csv", ","); // Load The Testing Data


		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	@AfterClass
	public static void tearDownAfterClass() throws Exception {
		System.out.println("tearDownAfter called");

	}

	@Before
	public void setUp() throws Exception {
		System.out.println("setUp called");
		
		 // find the nearest Centroids Methods
		Map<String,INDArray> result = Centroids.findNearestCentroids(Data, this.initial_centroids); // Call nearestCentroids() Method

		INDArray minimumDistance = result.get("minDist"); // Get the minimumDistance Array from List
		this.minimumDistanceIndex = result.get("minIdx"); // Get the Minimum Index Array from List
	}

	@After
	public void tearDown() throws Exception {
		System.out.println("tearDown called");

	}
    /**
     * In This Methods we will Implement the code from which we are going to test the 
     * Implementation of nearestCentroids() Method of KMeanMethod Class.
     */
	@Test
	public void test0NearestCentroids() {
		System.out.println("Test NearestCentroids called");
		// we will going to Check the nearestCentroids() method here
 
		  INDArray desiredResult = Nd4j.create(new double[] {0,2,1}).castTo(DataType.INT64); // The Result we want
		  
		  INDArray actualResult = this.minimumDistanceIndex.get(Nd4j.create(new double[] {0,1,2})); // The Result we actually get from Method
		
		  assertEquals("Both array Should be Equal",desiredResult, actualResult); // Compare Result
		  
	}

    /**
     * In this Method We will Implement the code which is going to test the calibrateCentroids() Methods of 
     * Centroids Class.
     */
	@Test
	public void tes2CalibrateCentroids() {
		System.out.println("Test Calibrate Centroids");
		
		INDArray newCentroids = Centroids.calibrateCentroids(Data, this.minimumDistanceIndex, this.numCentroids);
		
		INDArray desiredResult = Nd4j.create(new double[][] {{2.428301, 3.157924}, 
			                                                 {5.813503, 2.633656},
			                                                 {7.119387, 3.616684}});
		
		assertEquals("Both Array should be Equal ",desiredResult,newCentroids);


	}

	/**
	 * In This Method we are Testing the initialization of KMeans Algo
	 *
	 */
	@Test
	public void test3KMeansAlgo(){
		System.out.println("testKMeansAlgo Called");
		Map<String,INDArray> result =KMeans.initKMeans(Data,initial_centroids,10,numCentroids);

		INDArray actualNewCentroids = result.get("Centroids");
		INDArray actualMinIdx = result.get("minIdx").get(Nd4j.create(new double[] {0,1,2}));

		INDArray desiredNewCentroids = Nd4j.create(new double[][] {{1.95399466, 5.02557006},
				                                                {3.04367119, 1.01541041},
				                                                {6.03366736, 3.00052511}});
		INDArray desiredMinIdx = Nd4j.create(new double[] {0,2,2}).castTo(DataType.INT64);

		assertEquals("Both Result Should be Equal",desiredNewCentroids,actualNewCentroids);
		assertEquals("Both Arrays should be equal",desiredMinIdx,actualMinIdx);

	}

}
