package libsvm;

import java.util.Arrays;

public class svm_problem implements java.io.Serializable
{
	public int l;
	public double[] y;
	public svm_node[][] x;
	@Override
	public String toString() {
		return "svm_problem: l=" + l + "\ny=" + Arrays.toString(y) + ", \nx=" + Arrays.toString(x) + "]";
	}
	
}
