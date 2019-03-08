import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.StringTokenizer;

import libsvm.svm;
import libsvm.svm_model;
import libsvm.svm_node;

public class SVMClassifier {
	

	private svm_model model;
	private double[] measuredFeatures;
	
	private static int MAX_X = 30, MAX_Y = 20; //coordinates: X: 0-30  and  Y: 0-30

	//Scaling parameters
	private int max_index;

	private double lower = -1.0;
	private double upper = 1.0;
	private double y_lower;
	private double y_upper;
	private boolean y_scaling = false;

	private double y_max = -Double.MAX_VALUE;
	private double y_min = Double.MAX_VALUE;

	private double[] feature_max;
	private double[] feature_min;
	

	public static void main(String[] args) {
		System.out.println("*********************LOADING PARAMETERS***************************");
		SVMClassifier classifier = new SVMClassifier("model.dat", "scalingParameters.dat");

		System.out.println("*********************CLASSIFYING MEASURED DATA***************************");
		double[] measuredFeatures = { -72, 0, -100, -92, 0, -93 };

		classifier.setMeasuredFeatures(measuredFeatures);
		double predicted = classifier.predict();
		System.out.println("Predicted X = " + ((int) predicted) / MAX_Y + "Predicted Y = " + (int) predicted % MAX_Y);
	}

	public SVMClassifier(String modelFileName, String scalingParametersFileName) {

		try {
			model = svm.svm_load_model(modelFileName);
			System.out.println("Parameters = " + model.param);
			loadScalingParameters(scalingParametersFileName);
			
		} catch (IOException e) {
			System.out.println("ERROR reading model file " + modelFileName);
		}

	}

	public  double predict() {
		scaleData();
	    svm_node[] nodes = new svm_node[this.measuredFeatures.length];
	    for (int i = 0; i < this.measuredFeatures.length; i++) {
	        svm_node node = new svm_node();
	        node.index = i + 1;
	        node.value = this.measuredFeatures[i];
	        nodes[i] = node;
	    }
     
	    int[] labels = new int[model.nr_class];
	    svm.svm_get_labels(model,labels);

	    double[] prob_estimates = new double[model.nr_class];
	    double v = svm.svm_predict_probability(model, nodes, prob_estimates);

	    
	    return v;
	}

	private void scaleData() {
		
		System.out.println("measured data: ");
		for (int index = 0; index < measuredFeatures.length; index++) {
			System.out.print(measuredFeatures[index] + " ");
		}
		System.out.println();
		System.out.println("measured and scaled: ");
		for (int index = 0; index < measuredFeatures.length; index++) {
			/* skip single-valued attribute */
			if (Math.abs(feature_max[index] - feature_min[index]) < 1e-2)
				return;

			if (Math.abs(measuredFeatures[index] - feature_min[index]) < 1e-2)
				measuredFeatures[index] = lower;
			else if (Math.abs(measuredFeatures[index] - feature_max[index]) < 1e-2)
				measuredFeatures[index] = upper;
			else {
				measuredFeatures[index] = lower + (upper - lower) * (measuredFeatures[index] - feature_min[index])
						/ (feature_max[index] - feature_min[index]);
			}
			System.out.print(measuredFeatures[index] + " ");
		}
		System.out.println();

	}

	
	
	public double[] getMeasuredFeatures() {
		return measuredFeatures;
	}

	public void setMeasuredFeatures(double[] measuredFeatures) {
		measuredFeatures = new double[measuredFeatures.length];
		System.arraycopy(measuredFeatures, 0, this.measuredFeatures, 0, measuredFeatures.length);
		
	}

	private void loadScalingParameters(String scalingParametersFileName) {
		try {
			BufferedReader fp_restore = null;
			String restore_filename = scalingParametersFileName;
			if (restore_filename != null) {
				int idx, c;

				try {
					fp_restore = new BufferedReader(new FileReader(restore_filename));
				} catch (Exception e) {
					System.err.println("can't open file " + restore_filename);
					System.exit(1);
				}
				if ((c = fp_restore.read()) == 'y') {
					fp_restore.readLine();
					fp_restore.readLine();
					fp_restore.readLine();
				}
				fp_restore.readLine();
				fp_restore.readLine();

				String restore_line = null;
				while ((restore_line = fp_restore.readLine()) != null) {
					StringTokenizer st2 = new StringTokenizer(restore_line);
					idx = Integer.parseInt(st2.nextToken());
					max_index = Math.max(max_index, idx);
				}
				fp_restore = rewind(fp_restore, restore_filename);

			}
			try {
				feature_max = new double[(max_index + 1)];
				feature_min = new double[(max_index + 1)];
			} catch (OutOfMemoryError e) {
				System.err.println("can't allocate enough memory");
				System.exit(1);
			}

			for (int i = 0; i <= max_index; i++) {
				feature_max[i] = -Double.MAX_VALUE;
				feature_min[i] = Double.MAX_VALUE;
			}
			if (restore_filename != null) {
				int idx, c;
				// fp_restore rewinded in finding max_index
				double fmin, fmax;

				fp_restore.mark(2); // for reset
				if ((c = fp_restore.read()) == 'y') {
					fp_restore.readLine(); // pass the '\n' after 'y'
					StringTokenizer st = new StringTokenizer(fp_restore.readLine());
					y_lower = Double.parseDouble(st.nextToken());
					y_upper = Double.parseDouble(st.nextToken());
					st = new StringTokenizer(fp_restore.readLine());
					y_min = Double.parseDouble(st.nextToken());
					y_max = Double.parseDouble(st.nextToken());
					y_scaling = true;
				} else
					fp_restore.reset();

				if (fp_restore.read() == 'x') {
					fp_restore.readLine(); // pass the '\n' after 'x'
					StringTokenizer st = new StringTokenizer(fp_restore.readLine());
					lower = Double.parseDouble(st.nextToken());
					upper = Double.parseDouble(st.nextToken());
					System.out.println("x: " + lower + "-" + upper);
					String restore_line = null;
					while ((restore_line = fp_restore.readLine()) != null) {
						StringTokenizer st2 = new StringTokenizer(restore_line);
						idx = Integer.parseInt(st2.nextToken());
						fmin = Double.parseDouble(st2.nextToken());
						fmax = Double.parseDouble(st2.nextToken());
						if (idx <= max_index) {
							feature_min[idx] = fmin;
							feature_max[idx] = fmax;

							System.out.println(idx + ": " + feature_min[idx] + "-" + feature_max[idx]);
						}
					}
				}
				fp_restore.close();
			}
		} catch (IOException e) {
			System.out.println("ERROR reading the scaling parameters from " + scalingParametersFileName);
		}
	}

	private static BufferedReader rewind(BufferedReader fp, String filename) throws IOException {
		fp.close();
		return new BufferedReader(new FileReader(filename));
	}

}
