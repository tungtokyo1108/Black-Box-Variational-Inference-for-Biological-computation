package jp.ac.utokyo.biobigdata.example;

import java.util.Scanner;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.List;
import java.io.*;

public class DNACount {

    private static Scanner scanner = new Scanner(System.in);
    private static List<Character> REAL_NUCLEOTIDES = Arrays.asList('A', 'C', 'G', 'T');

    public void Counting (String input) throws FileNotFoundException, IOException{
            FileReader fileReader = new FileReader(input);
            BufferedReader bufferedReader = new BufferedReader(fileReader);
            String dna = bufferedReader.readLine();

            if (dna != null)
            {
                System.out.println("DNA sequence: \n" + dna);
            }

            Scanner sc = new Scanner(System.in);
            System.out.println("Please select the type of data structure: 0: Hash_Map or 1: Common");
            int c = sc.nextInt();

            if (c == 0) 
            {
                HashMap<Character, Integer> output = new HashMap<>();
                for (char nuc : REAL_NUCLEOTIDES) {
                    output.put(nuc, 0);
                }   

                for (int i=0; i < dna.length(); i++) {
                    char currentNuc = dna.charAt(i);
                    Integer currentNucCount = output.putIfAbsent(currentNuc, 1);
                    if (currentNucCount != null)
                    {
                        output.replace(currentNuc, currentNucCount + 1);
                    }
                }
                System.out.println("\nThis is counting DNA nucleotides program using Hash Map:");
                System.out.println(output);
            }
            else
            {
                int A = 0;
                int C = 0;
                int T = 0;
                int G = 0;

                int len = dna.length();

                for (int i=0; i<len; i++)
                {
                    if (dna.charAt(i) == 'A') A++;
                    else if (dna.charAt(i) == 'C') C++;
                    else if (dna.charAt(i) == 'T') T++;
                    else G++;
                }
                System.out.println("\nThis is couting DNA nucleotide program using common method");
                System.out.println("A: " + A + "\n" + "C: " + C + "\n" + "T: " + T + "\n"+ "G: " + G);
            }

            bufferedReader.close();
    }

    public void findingMotif (String DNAinput, String motifinput) throws FileNotFoundException, IOException{

        FileReader fileReader1 = new FileReader(DNAinput);
        BufferedReader bufferedReader1 = new BufferedReader(fileReader1);
        String DNA = bufferedReader1.readLine();
        System.out.println("DNA sequence: \n" + DNA);

        FileReader fileReader2 = new FileReader(motifinput);
        BufferedReader bufferedReader2 = new BufferedReader(fileReader2);
        String motif = bufferedReader2.readLine();
        System.out.println("Motif sequence: \n" + motif);

        int dna_length = DNA.length();
        int motif_length = motif.length();
        System.out.println("The DNA sequence contains the motif at positions: ");
        for (int i=0; i < dna_length-motif_length; i++) {
            String subseq = DNA.substring(i, i+motif_length);
            if (subseq.equals(motif)) {
                System.out.print((i+1) + " ,");
            }
        }

        bufferedReader1.close();
        bufferedReader2.close();
    }

    public static void main(String[] args) {

        System.out.println("Please enter the filename: ");
        String filename1 = scanner.nextLine();
        String filename2 = scanner.nextLine();
        DNACount test = new DNACount();

        try {
            //test.Counting(filename1);
            test.findingMotif(filename1, filename2);
        } catch(FileNotFoundException ex) {
            System.out.println("Unable to open file");
        } catch(IOException ex) {
            System.out.println("Error reading file");
        }
    }
}
