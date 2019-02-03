package tud.ke.ml.project.classifier;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import tud.ke.ml.project.util.Pair;
import weka.core.Instance;

/**
 * This implementation assumes the class attribute is always available (but
 * probably not set).
 */
public class NearestNeighbor extends AbstractNearestNeighbor implements Serializable {
	private static final long serialVersionUID = 8662234558169046563L;

	protected double[] scaling;
	protected double[] translation;
	protected List<List<Object>> data;

	@Override
	public String getMatrikelNumbers() {
		String matriculationnumber = "2912264,2002954";
		System.out.println(matriculationnumber);
		return matriculationnumber;
	}

	@Override
	protected void learnModel(List<List<Object>> data) {
		this.data = data;
	}

	@Override
	protected Object vote(List<Pair<List<Object>, Double>> subset) {
		// every class with it's distance as double, e.g. "yes"=2.4, "no"=4.5

		// create the boolean object inverse that determines which votes are calculated
		boolean inverse = isInverseWeighting();

		if (inverse == false) {
			Map<Object, Double> votes = getUnweightedVotes(subset);
			return getWinner(votes);
		} else {
			Map<Object, Double> votes = getWeightedVotes(subset);
			return getWinner(votes);
		}
	}

	@Override
	protected Map<Object, Double> getUnweightedVotes(List<Pair<List<Object>, Double>> subset) {
		Map<Object, Double> myClasses = new HashMap<Object, Double>();
		// return classes with their number of votes?
		for (Pair pair : subset) {
			List<String> listOfAttributes = (List<String>) (pair.getA());

			// key=classname, value number of samples with same key
			if (myClasses.containsKey(pair.getA()))
				myClasses.put(listOfAttributes.get(listOfAttributes.size() - 1),
						(double) myClasses.get(pair.getA()) + 1);
			else {
				myClasses.put(listOfAttributes.get(listOfAttributes.size() - 1), 1.0);

			}
		}
		return myClasses;
	}

	@Override
	protected Map<Object, Double> getWeightedVotes(List<Pair<List<Object>, Double>> subset) {

		Map<Object, Double> myClasses = new HashMap<Object, Double>();
		// return classes with their number of votes?
		for (Pair pair : subset) {
			List<String> listOfAttributes = (List<String>) (pair.getA());

			// key=classname, value number of samples with same key
			if (myClasses.containsKey(pair.getA())) {
				// multiply with the weight which is saved in the pair
				System.out.println(
						"getweighted-classname=" + pair.getA().toString() + "-weight=" + pair.getB().toString());
				myClasses.put(listOfAttributes.get(listOfAttributes.size() - 1),
						(double) myClasses.get(pair.getA()) + 1 * ((double) pair.getB()));
			} else {
				myClasses.put(listOfAttributes.get(listOfAttributes.size() - 1), 1.0 * ((double) pair.getB()));

			}
		}
		return myClasses;
	}

	@Override
	protected Object getWinner(Map<Object, Double> votes) {
		Map.Entry<Object, Double> myEntry = Collections.max(votes.entrySet(),
				Comparator.comparingDouble(Map.Entry::getValue));

		return myEntry.getKey();
	}

	@Override
	protected List<Pair<List<Object>, Double>> getKNearest(List<Object> testData) {
		// raw data is passed and a list of pairs of instances with their distances is
		// returned
		List<Pair<List<Object>, Double>> listOfInstanceDistancePairs = new ArrayList<Pair<List<Object>, Double>>();

		int metric = getMetric();
		// loop over all instances in the training data and determine the distance to
		// the passed instance
		for (List<Object> instance2 : this.data) {
			if (metric == 0) {
				double distance = determineManhattanDistance(testData, instance2);
				listOfInstanceDistancePairs.add(new Pair(instance2, distance));
			} else {
				double distance = determineEuclideanDistance(testData, instance2);
				listOfInstanceDistancePairs.add(new Pair(instance2, distance));
			}
		}

		// return only k instances: sort by lowest distance and return k elements
		Collections.sort(listOfInstanceDistancePairs, new Comparator<Pair>() {
			@Override
			public int compare(Pair left, Pair right) {
				return Double.compare((double) left.getB(), (double) right.getB());
			}
		});

		/*
		 * return list of k pairs of the instances in the training data with their
		 * distance to the passed instance
		 */
		return listOfInstanceDistancePairs.subList(0, getK());
	}

	@Override
	protected double determineManhattanDistance(List<Object> instance1, List<Object> instance2) {
		double distance = 0.0;
		// excluding the class-attribute
		for (int i = 0; i < getClassAttributeIndex() - 1; i++) {

			// attribute i
			Object attrOfInstance1 = instance1.get(i);
			Object attrOfInstance2 = instance2.get(i);
			// if attribute i of instance 1 equals that of instance 2 than distance is 0
			// else 1
			try
			{
				double value1=Double.parseDouble(attrOfInstance1.toString());
				double value2=Double.parseDouble(attrOfInstance2.toString());
				distance+=(value1-value2);

			}
			// if it is a nominal value then just add 1 if the two values are equal
			catch(NumberFormatException e)
			{
			if (!attrOfInstance1.toString().equals(attrOfInstance2.toString()))
				distance += 1;
			}

		}
		return distance;
	}

	@Override
	protected double determineEuclideanDistance(List<Object> instance1, List<Object> instance2) {

		double distance = 0.0;
		// excluding the class-attribute
		for (int i = 0; i < getClassAttributeIndex() - 1; i++) {

			// attribute i
			Object attrOfInstance1 = instance1.get(i);
			Object attrOfInstance2 = instance2.get(i);
			// if attribute i of instance 1 equals that of instance 2 than distance is 0
			// else calculate the euclidean Distance between this points
			boolean isNumber=true;
			try
			{
				double value1=Double.parseDouble(attrOfInstance1.toString());
				double value2=Double.parseDouble(attrOfInstance2.toString());
				distance+=(value1-value2)*(value1-value2);

			}
			catch(NumberFormatException e)
			{
				// if attribute i of instance 1 equals that of instance 2 than distance is 0
				// else 1
				if (!attrOfInstance1.toString().equals(attrOfInstance2.toString()))
					distance += 1;
			}
		}
	
		distance = Math.sqrt(distance);

		return distance;

	}

	@Override
	protected double[][] normalizationScaling() {
		/*
		 * first the min and max value of each attribute from the training data has to be calculated
		 * then the scaling factor has to be saved as max-min and the tranlating factor is just min
		 * When normalizing each value is normalized with: (v - min)/(max-min)
		 */
		int numberOfAttributesAndClass=this.data.get(0).size();
		
		double[][] scales=new double[numberOfAttributesAndClass][2];
		Object[][] valuesMatrix=new Object[numberOfAttributesAndClass][this.data.size()];
		int i=0;
		int j=0;
		//save values a appropriately structured matrix to calculate the min and max value
		for(List<Object> listOfObj:this.data)
		{
			for(Object obj:listOfObj)
			{
				valuesMatrix[j][i]=obj;
						j++;
			}
		}
		//TODO: writing the scaling and translating factor to array and returning it
		/*
		for(int k=0;k<numberOfAttributesAndClass;k++)
		{
			//writing the scaling factor: (max - min)
			valuesMatrix[k][0]=Collections.max(valuesMatrix[k],new Comparator<Object>() {

				@Override
				public int compare(Object o1, Object o2) {
					double d1= Double.parseDouble(o1.toString());
					double d2=Double.parseDouble(2.toString());
				    if (d1==d2)
				    {
			            return 0;
			        } else if (d1 > d2) {
			            return -1;
			        } else if (d1 < d2) {
			            return 1;
			        }
				
					// TODO Auto-generated method stub
					return 0;
			
			
		}});
			}
			*/
		return new double[][] { { 0.0 } };
	}
}
