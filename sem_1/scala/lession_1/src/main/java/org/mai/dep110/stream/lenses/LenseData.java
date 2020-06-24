package org.mai.dep110.stream.lenses;

/**
 * Created by Asus on 10/7/2018.
 *
 * dataset description - https://archive.ics.uci.edu/ml/datasets/Lenses
 */
public class LenseData {

    /*
    * -- 3 Classes
    *    1 : the patient should be fitted with hard contact lenses,
    *    2 : the patient should be fitted with soft contact lenses,
    *    3 : the patient should not be fitted with contact lenses.
     */
    private int lenseClass;

    /*
    * age of the patient: (1) young, (2) pre-presbyopic, (3) presbyopic
    */
    private int patientAge;

    /*
    * spectacle prescription: (1) myope, (2) hypermetrope
    */
    private int spectaclePrescription;

    /*
    *  astigmatic: (1) no, (2) yes
    */
    private boolean isAstigmatic;

    /*
    * tear production rate: (1) reduced, (2) normal
    */
    private int tearProductionRate;

    public LenseData(int lenseClass, int patientAge, int spectaclePrescription, boolean isAstigmatic, int tearProductionRate) {
        this.lenseClass = lenseClass;
        this.patientAge = patientAge;
        this.spectaclePrescription = spectaclePrescription;
        this.isAstigmatic = isAstigmatic;
        this.tearProductionRate = tearProductionRate;
    }

    public static LenseData parse(String line) {
        String[] parts = line.split(",");

        return new LenseData(
                Integer.parseInt(parts[0]),
                Integer.parseInt(parts[1]),
                Integer.parseInt(parts[2]),
                Integer.parseInt(parts[3]) == 1 ? true : false,
                Integer.parseInt(parts[4])
        );
    }

    public int getLenseClass() {
        return lenseClass;
    }

    public int getPatientAge() {
        return patientAge;
    }

    public int getSpectaclePrescription() {
        return spectaclePrescription;
    }

    public boolean isAstigmatic() {
        return isAstigmatic;
    }

    public int getTearProductionRate() {
        return tearProductionRate;
    }

    @Override
    public String toString() {
        return "LenseData{" +
                "lenseClass=" + lenseClass +
                ", patientAge=" + patientAge +
                ", spectaclePrescription=" + spectaclePrescription +
                ", isAstigmatic=" + isAstigmatic +
                ", tearProductionRate=" + tearProductionRate +
                '}';
    }
}
