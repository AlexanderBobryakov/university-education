package org.mai.dep110.stream.employees;

public class Employee {
    private String name;
    private Profession profession;
    private Position position;
    private SEX sex;
    private int age;
    private Double salary;

    public Employee(String name, Profession profession, Position position, SEX sex, int age, Double salary) {
        this.name = name;
        this.profession = profession;
        this.position = position;
        this.sex = sex;
        this.age = age;
        this.salary = salary;
    }

    public String getName() {
        return name;
    }

    public Profession getProfession() {
        return profession;
    }

    public Position getPosition() {
        return position;
    }

    public SEX isSex() {
        return sex;
    }

    public int getAge() {
        return age;
    }

    public Double getSalary() {
        return salary;
    }

    @Override
    public String toString() {
        return "Employee{" +
                "name='" + name + '\'' +
                ", profession=" + profession +
                ", position=" + position +
                ", sex=" + sex +
                ", age=" + age +
                ", salary=" + salary +
                '}';
    }

    public static Employee parse(String line) {
        String[] parts = line.split(",");

        return new Employee(
                parts[0],
                Profession.valueOf(parts[1]),
                Position.valueOf(parts[2]),
                SEX.valueOf(parts[3]),
                Integer.parseInt(parts[4]),
                Double.parseDouble(parts[5])
        );
    }

    public enum Profession{
        SOFTWARE_DEVELOPER, SYSTEM_ANALYTIC, BUSINESS_ANALYTIC, TESTER, ARCHITECT
    }

    public enum Position{
        JUNIOR, REGULAR, SENIOR, LEAD
    }

    public enum SEX {
        MALE, FEMALE
    }
}

