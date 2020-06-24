package org.mai.dep110.stream.employees;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Comparator;
import java.util.List;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Created by Asus on 10/8/2018.
 */
public class Main {
    public static void main(String[] args) throws IOException {
        //generateDataSet();

        Function<String, Employee> f = Employee::parse;

        List<Employee> employees = Files
                .lines(Paths.get("employees.data"))
                .map(f)
                .collect(Collectors.toList());

        //filter example
        employees
                .stream()
                .filter( e -> e.isSex() == Employee.SEX.FEMALE && e.getSalary() < 10.0d)
                .forEach(System.out::println);

        //aggregate example
        Double grossSalaryAverage = employees
                .stream()
                .mapToDouble(e -> e.getSalary()*100/87)
                .average()
                .getAsDouble();

        System.out.println("grossSalaryAverage = " + grossSalaryAverage);
        System.out.println();

        //group example
        employees
                .stream()
                .collect(
                        Collectors
                                .groupingBy(Employee::getPosition)
                )
                .entrySet()
                .forEach(System.out::println);

        System.out.println();

        //group with aggregate example
        employees
                .stream()
                .collect(
                        Collectors.groupingBy(
                                Employee::getProfession,
                                Collectors.minBy(Comparator.comparingInt(Employee::getAge))))
                .entrySet()
                .forEach(System.out::println);

    }

    static void generateDataSet() throws IOException {
        Files.lines(Paths.get("names.data")).map(Main::getEmployee).map(Main::getLine).forEach(System.out::println);
    }

    static Employee getEmployee(String name) {
        Employee.Profession profession = getProfession(Math.random());
        Employee.Position position = getPosition(Math.random());
        Employee.SEX sex = getSex(Math.random());
        int age = (int)(Math.random()*45+20);
        double salary = Math.random()*1000;

        String employeeName = name;
        if(sex == Employee.SEX.FEMALE)
            employeeName = employeeName+"а";

        return new Employee(employeeName, profession, position, sex, age, salary);
    }

    static String getLine(Employee e) {
        return Stream.of(e.getName(), e.getProfession().name(), e.getPosition().name(), e.isSex().name(), ""+e.getAge(), ""+e.getSalary()).reduce((s1, s2) -> s1 + "," + s2).get();
    }

    static Employee.Profession getProfession(double d) {
        if(d < 0.2d) { return Employee.Profession.ARCHITECT; }
        else if(d < 0.4d) { return Employee.Profession.BUSINESS_ANALYTIC; }
        else if(d < 0.6d) { return Employee.Profession.SOFTWARE_DEVELOPER; }
        else if(d < 0.8d) { return Employee.Profession.SYSTEM_ANALYTIC; }
        else{ return Employee.Profession.TESTER; }
    }

    static Employee.Position getPosition(double d) {
        if(d < 0.25d) { return Employee.Position.JUNIOR; }
        else if(d < 0.5d) { return Employee.Position.REGULAR; }
        else if(d < 0.75d) { return Employee.Position.SENIOR; }
        else{ return Employee.Position.LEAD; }
    }

    static Employee.SEX getSex(double d) {
        if(d < 0.5d) { return Employee.SEX.FEMALE; }
        else{ return Employee.SEX.MALE; }
    }

}
