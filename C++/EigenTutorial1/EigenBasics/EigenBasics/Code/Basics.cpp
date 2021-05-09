#include <iostream>
#include <stdio.h>
#include <Eigen/Dense>


using namespace std;
using namespace Eigen;


int main() {

	// example of Dynamic matrix we use this type of matrix when we do not know the exact size of the matrix for ex- during reading data 
	// from a csv file etc. so at that time we use this type of matix.

  //Declare Dynamic Matrix

	MatrixXd dynamicType; // here Capital 'X' means the size of the matrix is Unknown (Dynamic Matrix) and 'd' stands for data type double 
						  // we can also initialize this matrix here with given size ex- MatrixXd dynamicType(6,8) like this.

	// or

	Matrix<double, Dynamic, Dynamic> dynamicType2; // here first parameter represent data type second and Third parmeter represents row size and column size

	// fix size square matrix -- 3*3 , we uses this type of matrix when we know the size of the matrix we can not resize this type of 
	// matrix

	Matrix3d fixedSize; // here 3 stands for square matrix size and 'd' for data type.

  //or

	Matrix<float, 5, 5> fixedSize2;

	// How to initialize the array

	//for fixed size matrix

	fixedSize << 1, 2, 3,
		4, 5, 6,
		7, 8, 9;

	cout << "fixedSize = \n" << fixedSize << endl;

	// initializing randomly

	fixedSize = Matrix3d::Random(); // it will generate 3*3 random matrix. Here we didn't pass the size of the matrix since we know 
									// that size of the matrix is 3*3.

	cout << "fixed Size Random = \n" << fixedSize << endl;

	//initializing with Constant number

	fixedSize = Matrix3d::Constant(1.0); // it will put every number equal to one

	cout << "Fixed Size with Constant = \n" << fixedSize << endl;



	// Now look at How we should initialize Dynamic randomly

	dynamicType = MatrixXd::Random(4, 3); // Here we have passed the size 4,3 because it is a dynamic array.

	cout << "Dynamic Type randomly = \n" << dynamicType << endl;

	// for Dynamic Arrays we can use Constant() function for initializing and setting constant for both

	dynamicType = MatrixXd::Constant(6, 3, 2); // here 6 is the number of rows 3 is Column and 2 is Constant

	cout << "Dynamic Matrix with Constant = \n" << dynamicType << endl;

	// Genrating Zero matrix

	Matrix3d zeros = Matrix3d::Zero();

	cout << "Zero Matrix = \n" << zeros << endl;

	// Genrating Ones Matrix

	MatrixXd onesMat = MatrixXd::Ones(6, 8);

	cout << "Ones Matrix = \n" << onesMat << endl;

	// Declaring a Vector

	VectorXd vecDynamic; // or 
	Matrix<double, Dynamic, 1> vecDynXd;

	Vector3d vecFixed; // or
	Matrix<double, 3, 1> vec3d;

	// now we can initialize both type as same as before but keeping in mind that they are Vectors.











}