import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.StringTokenizer;
import java.util.concurrent.ThreadLocalRandom;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;
import libsvm.svm_parameter;
import libsvm.svm_problem;

//https://github.com/cjlin1/libsvm/blob/master/README
//https://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf
public class SVMTrainer {
	private String inputFileName;
	private String resultFileName;
	private String modelFileName;
	private svm_problem problem;
	private svm_parameter parameters;
	private svm_model model;
	private double[] testingSetLabels;
	private double[][] testingSetData;
	
	private static int MAX_X = 30, MAX_Y = 20; //coordinates: X: 0-30  and  Y: 0-30

	// CROSS VALIDATION TO FIND C and GAMMA
	private static int FOLD = 5;
	// Coarse grid search
	//private static int C_BEGIN = -5, C_END = 15, C_STEP = 2;// c_range = 2^{begin,...,begin+k*step,...,end}
	//private static int G_BEGIN = -15, G_END = 3, G_STEP = 2;// g_range = 2^{begin,...,begin+k*step,...,end}
	//*******C=-3, g=1, accuracy=27.317983820784068
	//*******C=-3, g=3, accuracy=30.77162414436839
	//*******C=-1, g=1, accuracy=36.46546359676416
	//*******C=-1, g=3, accuracy=44.99066583696328
	//Fine grid search
	private static int C_BEGIN = 2, C_END = 8; 
	private static double C_STEP = 2;// c_range = 2^{begin,...,begin+k*step,...,end}
	private static int G_BEGIN = 3, G_END = 5; 
	private static double G_STEP = 0.25;// g_range = 2^{begin,...,begin+k*step,...,end}

	
	private static final int SVM_PROBABILITY = 1;
	private static final double SVM_NU = 0.5;
	private static final int SVM_CACHE_SIZE = 100000;
	private static final double SVM_EPS = 0.001;

	public static void main(String[] args) {
		SVMTrainer trainer = new SVMTrainer("measurements.csv", "Results.txt", "model.dat");
		trainer.train();
	}

	public SVMTrainer(String inputFileName, String resultFileName, String modelFileName) {
		super();
		this.inputFileName = inputFileName;
		this.resultFileName = resultFileName;
		this.modelFileName = modelFileName;
	}

	public void train() {
		problem = new svm_problem();

		// 1. load data from csv file,
		// 2. scale, and transform data to the format of an SVM package
		System.out.println("*********************LOADING AND SCALING DATA***************************");
		loadAndScaleData();

		// 3.Load parameters
		System.out.println("*********************LOADING PARAMETERS***************************");
		loadParameters();

		// 4. Use cross-validation to find the best parameter C and γ
		// C > 0 is the penalty parameter of the error term
		// γ is a kernel parameter
		// Cross-validation is a technique for evaluating ML models by training
		// several ML models on subsets of the available input data
		// and evaluating them on the complementary subset of the data.
		// In k-fold cross-validation, you split the input data into k subsets
		// of data (also known as folds).
		// You train an ML model on all but one (k-1) of the subsets, and then
		// evaluate the model on the subset that was not used
		// for training. This process is repeated k times, with a different
		// subset reserved for evaluation (and excluded from training) each
		// time.

		System.out.println("*********************USING CROSS-VALIDATION TO FIND BEST C and GAMMA USING THE TRAINING SET***************************");
		findandLoadBestCAndGamma();

		// 5. Use the best parameter C and γ to train the whole training set

		System.out.println("*********************TRAINING THE MACHINE WITH BEST C and GAMMA USING THE TRAINING SET***************************");
		model = svm.svm_train(problem, parameters);
		try {
			svm.svm_save_model(modelFileName, model);

			// 6. Evaluate generated model

			System.out.println("*********************EVALUATING THE GENERATED MODEL USING TESTING SET***************************");
			evaluate();
		} catch (IOException e) {
			System.out.println("ERROR while saving the model to the file:" + modelFileName);
		}

	}

	private void loadAndScaleData() {
		try {

			// 1. read original file
			FileReader fileReader = new FileReader(inputFileName);
			BufferedReader bufferedReader = new BufferedReader(fileReader);
			String line = null;
			// 2. write to intermediate format (in libsvm format) for scaling
			// The format of training and testing data file is:
			// <label> <index1>:<value1> <index2>:<value2> ...
			FileWriter fw = new FileWriter(new File("measurements.dat"));

			while ((line = bufferedReader.readLine()) != null) {
				StringTokenizer st = new StringTokenizer(line, ",");
				int X = Integer.parseInt(st.nextToken());
				int Y = Integer.parseInt(st.nextToken());
				int label = X * MAX_Y + Y;
				fw.write(label + " ");
				int ind = 1;
				while (st.hasMoreTokens()) {
					String token = st.nextToken();
					fw.write(String.valueOf(ind) + ":" + token + " ");
					ind++;
				}
				fw.write("\n");
			}
			bufferedReader.close();
			fw.close();
			System.out.println("  ****data formatted into measurements.dat");

			// 3. scale (results will be written in output file scaledMeasurements.dat")
			// String[] argvs = {"-y", "-1", "1", "-o",
			// "scaledMeasurements.dat", "measurements.dat"};
			String[] argvs = { "-s", "scalingParameters.dat", "-o", "scaledMeasurements.dat", "measurements.dat" };
			svm_scale.main(argvs);
			System.out.println("  ****data scaled into scaledMeasurements.dat");

			
			
			// 4.read and load scaled file into inputData and labels
			FileReader fileReader2 = new FileReader("scaledMeasurements.dat");
			BufferedReader bufferedReader2 = new BufferedReader(fileReader2);
			String line2 = null;

			List<List<Double>> inputData = new ArrayList<List<Double>>();
			List<Double> labels = new ArrayList<Double>();
			while ((line2 = bufferedReader2.readLine()) != null) {

				StringTokenizer st = new StringTokenizer(line2, " :");
				Double label = Double.parseDouble(st.nextToken());
				//System.out.print("label = " + label+":");
				labels.add(label);
				List<Double> dataPoint = new ArrayList<Double>();

				int i = 1;
				while (st.hasMoreTokens()) {
					int inde = Integer.parseInt(st.nextToken());
					if (inde != i) {
						for (int j = i; j < inde; j++) {
							dataPoint.add(0.0);
							i++;
						}
					}
					Double token = Double.parseDouble(st.nextToken());
					dataPoint.add(token);
					//System.out.print(" " + token);
					i++;
				}
				inputData.add(dataPoint);
				//System.out.println();

			}
			bufferedReader2.close();
			System.out.println("  ****data read from scaledMeasurements.dat. Num data: "+inputData.size());

			// 5. shuffle data points
			Random rnd = ThreadLocalRandom.current();
			for (int i = labels.size() - 1; i > 0; i--) {
				int index = rnd.nextInt(i + 1);
				Double label = labels.get(index);
				labels.set(index, labels.get(i));
				labels.set(i, label);

				List<Double> datapoint = inputData.get(index);
				inputData.set(index, inputData.get(i));
				inputData.set(i, datapoint);
			}
			System.out.println("  ****data shuffled");

			// 6. Partition into training set (problem, size = 9/10) and test set
			// (1/10)
			// Training set
			int trainingSetCount = labels.size() * 9 / 10;
			problem.l = trainingSetCount;
			problem.x = new svm_node[trainingSetCount][];
			problem.y = new double[trainingSetCount];

			for (int i = 0; i < trainingSetCount; i++) {
				problem.x[i] = new svm_node[inputData.get(i).size()];
				for (int j = 1; j <= inputData.get(i).size(); j++) {
					svm_node node = new svm_node();
					node.index = j;
					node.value = inputData.get(i).get(j - 1);
					problem.x[i][j - 1] = node;
				}
				problem.y[i] = labels.get(i).doubleValue();
			}
			System.out.println("  ****training set (9/10 of data = " + trainingSetCount + ") associated with problem");
			
			// Test set
			int testingSetCount = labels.size() - trainingSetCount;
			testingSetLabels = new double[testingSetCount];
			testingSetData = new double[testingSetCount][];
			int ii = 0;
			for (int i = trainingSetCount; i < labels.size(); i++) {
				testingSetData[ii] = new double[inputData.get(i).size()];
				for (int j = 0; j < inputData.get(i).size(); j++) {
					testingSetData[ii][j] = inputData.get(i).get(j);
				}
				testingSetLabels[ii] = labels.get(i).doubleValue();
				ii++;
			}
			System.out.println("  ****testing set (1/10 of data = " + testingSetCount + ") ready");

			/*
			 * for(int i = 0; i < problem.x.length; i ++){
			 * System.out.print("i="+i+", label="+problem.y[i]+ ", values=");
			 * for(int j = 0; j < problem.x[i].length; j ++){
			 * System.out.print(problem.x[i][j]+", "); } System.out.println(); }
			 */

		} catch (FileNotFoundException ex) {
			System.out.println("Error reading file '" + inputFileName + "'. File does not exist.");
		} catch (IOException ex) {
			System.out.println("Error reading file '" + inputFileName + "'");
		}

	}

	private void loadParameters() {

		parameters = new svm_parameter();
		parameters.probability = SVM_PROBABILITY;
		parameters.nu = SVM_NU;
		parameters.svm_type = svm_parameter.C_SVC;//C_SVC  NU_SVC  ONE_CLASS EPSILON_SVR NU_SVR

		// KERNEL: training vectors xi are mapped into a higher (maybe infinite)
		// dimensional space
		// by the kernel function.
		// Then SVM finds a linear separating hyperplane with the maximal margin
		// in this higher dimensional space.
		// RBF kernel K(x, y) = e−γ||x−y||2
		parameters.kernel_type = svm_parameter.RBF; //LINEAR POLY RBF SIGMOID PRECOMPUTED
		parameters.cache_size = SVM_CACHE_SIZE;
		parameters.eps = SVM_EPS;

		System.out.println("  ****"+ parameters);

	}
	
	private void findandLoadBestCAndGamma() {
		double maxAccuracy = 0;
		double CAtMaxAccuracy = C_BEGIN;
		double GAtMaxAccuracy = G_BEGIN;
		FileWriter fileWriter;
		try {
			fileWriter = new FileWriter(new File(resultFileName));

			for (double c = C_BEGIN; c <= C_END; c += C_STEP) {
				parameters.C = Math.pow(2, c);
				for (double g = G_BEGIN; g <= G_END; g += G_STEP) {
					parameters.gamma = Math.pow(2, g);
					double[] predictedLabel = new double[problem.l];
					svm.svm_cross_validation(problem, parameters, FOLD, predictedLabel);
					double accuracy = getAccuracy(problem.y, predictedLabel);
					System.out.println(
							"*******C=" + c + ", g=" + g + ", accuracy=" + accuracy + ", parameters=" + parameters);
					fileWriter.write("*******C=" + c + ", g=" + g + ", accuracy=" + accuracy + ", parameters="
							+ parameters + "\n");
					if (accuracy > maxAccuracy) {
						maxAccuracy = accuracy;
						CAtMaxAccuracy = c;
						GAtMaxAccuracy = g;
					}
				}
			}

			parameters.C = Math.pow(2, CAtMaxAccuracy);
			parameters.gamma = Math.pow(2, GAtMaxAccuracy);
			System.out.println(
					"******MAX******C=" + CAtMaxAccuracy + ", g=" + GAtMaxAccuracy + ", accuracy=" + maxAccuracy + "%");
			fileWriter.write("******MAX******C=" + CAtMaxAccuracy + ", g=" + GAtMaxAccuracy + ", accuracy="
					+ maxAccuracy + "%" + "\n");
			fileWriter.flush();
			fileWriter.close();
		} catch (IOException e) {
			System.out.println("ERROR writing the results to the file:" + resultFileName);
		}
	}

	private void evaluate() {
		FileWriter fileWriter;
		try {
			fileWriter = new FileWriter(new File(resultFileName));

			int numErrors = 0;
			System.out.println("model.nr_class="+model.nr_class);
			for (int testInstance = 0; testInstance < testingSetLabels.length; testInstance++) {
				svm_node[] nodes = new svm_node[testingSetData[testInstance].length]; //vector of values
				for (int feature = 0; feature < testingSetData[testInstance].length; feature++) {
					svm_node node = new svm_node();
					node.index = feature + 1;
					node.value = testingSetData[testInstance][feature];
					nodes[feature] = node;
				}

				
				int[] labels = new int[model.nr_class];
				svm.svm_get_labels(model, labels);

				double[] prob_estimates = new double[model.nr_class];
				double v = svm.svm_predict_probability(model, nodes, prob_estimates);

				for (int j = 0; j < model.nr_class; j++) {
					 System.out.print("(" + labels[j] + ":" + prob_estimates[j] +
					 ")");
				}
				System.out.println("(Actual:" + testingSetLabels[testInstance] + " Prediction:" + v + ")");
				if (Math.abs(testingSetLabels[testInstance] - v) > 1e-2) {
					numErrors++;
					System.out.println("ERROR: " + testingSetLabels[testInstance] + " " + v);
				}
				
				
			}
			System.out.println("Error: " + (numErrors / (double) testingSetLabels.length * 100) + "%");
			System.out.println("Accuracy: " + (100 - (numErrors / (double) testingSetLabels.length * 100)) + "%");
			fileWriter.write("Error: " + (numErrors / (double) testingSetLabels.length * 100) + "%" + "\n");
			fileWriter.write("Accuracy: " + (100 - (numErrors / (double) testingSetLabels.length * 100)) + "%" + "\n");

			fileWriter.flush();
			fileWriter.close();
		} catch (IOException e) {
			System.out.println("Error writing to file '" + resultFileName + "'");
		}

	}



	private static double getAccuracy(double[] actualLabel, double[] predictedLabel) {
		double accuracy = 0;
		int numCorrect = 0;
		for (int i = 0; i < predictedLabel.length; i++) {
			if (predictedLabel[i] == actualLabel[i]) {
				numCorrect++;
			}
		}
		accuracy = numCorrect / (double) predictedLabel.length * 100;
		return accuracy;
	}

}
