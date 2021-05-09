import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * @author Akshay Kumar 08/05/21 2:37 AM
 * @project Regression
 **/
public class test {

    public static void main(String[] args) {
        INDArray stringArray = Nd4j.create("Hello","RRR");
        INDArray integerArray = Nd4j.create(new double[]{1,2});
        System.out.println(stringArray);
        System.out.println(integerArray);

        Map<String,INDArray> arrayList = new HashMap<>();
        arrayList.put("string",stringArray);
        arrayList.put("double",integerArray);


    }
}
