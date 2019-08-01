/**
 *  Big Data Technology
 *
 *  Created on: July 29, 2019
 *  Data Scientist: Tung Dang
 */


package jp.ac.utokyo.biobigdata.example;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Map.Entry;
import org.biojava.nbio.core.sequence.DNASequence;
import org.biojava.nbio.core.sequence.io.FastaReaderHelper;
import org.biojava.nbio.core.sequence.ProteinSequence;
import org.biojava.nbio.core.sequence.compound.AminoAcidCompound;
import org.biojava.nbio.core.sequence.compound.AminoAcidCompoundSet;
import org.biojava.nbio.core.sequence.io.FastaReader;
import org.biojava.nbio.core.sequence.io.GenericFastaHeaderParser;
import org.biojava.nbio.core.sequence.io.ProteinSequenceCreator;
import org.biojava.nbio.core.sequence.io.util.ClasspathResource;
import java.util.Scanner;
import java.io.IOException;
import java.io.InputStream;

public class FastaSubstrings {
    
    private static Scanner scanner = new Scanner(System.in);

    private void getSequence(String path) throws Exception {
        ClasspathResource r = new ClasspathResource(path);
        FastaReader<ProteinSequence, AminoAcidCompound> fastaReader = null;
        try (InputStream inStream = r.getInputStream()) 
        {
            fastaReader = new FastaReader<ProteinSequence, AminoAcidCompound>(
                inStream,
                new GenericFastaHeaderParser<ProteinSequence, AminoAcidCompound>(),
                new ProteinSequenceCreator(AminoAcidCompoundSet.getAminoAcidCompoundSet())
            );
            LinkedHashMap<String, ProteinSequence> sequences = fastaReader.process();
            System.out.println("The size of protein sequence" + sequences.size());
            System.out.println("The key of protein sequence" + sequences.containsKey("PO2768"));
            System.out.println("The length of protein sequenc" + sequences.get("PO2768").getLength());
        }
    }

    public void testgetSequence() throws Exception{
        getSequence("/home/tungutokyo/Desktop/AdvancedJava/biobigdata/target/P02768.fasta");
    }

    public void getSequence1(InputStream fastain, String outfile) throws IOException
    {
        boolean seqFound = false;
        Writer out = new BufferedWriter(new FileWriter(outfile));
        // LinkedHashMap<String, ProteinSequence> seqs = FastaReaderHelper.readFastaProteinSequence(fastain);
        FastaReader<ProteinSequence, AminoAcidCompound> fastaReader = null;
        fastaReader = new FastaReader<ProteinSequence, AminoAcidCompound>(
                fastain,
                new GenericFastaHeaderParser<ProteinSequence, AminoAcidCompound>(),
                new ProteinSequenceCreator(AminoAcidCompoundSet.getAminoAcidCompoundSet())
            );
        LinkedHashMap<String, ProteinSequence> seqs = fastaReader.process();
        System.out.println("The first check input data\n");
        System.out.println("The size of protein sequence: " + seqs.size());
        System.out.println("The key of protein sequence: " + seqs.containsKey("PO2768"));
        // System.out.println("The length of protein sequence: " + seqs.get("PO2768").getLength());
        for (Map.Entry<String,ProteinSequence> entry : seqs.entrySet())
        {
            String id = entry.getKey();
            String seq = entry.getValue().getSequenceAsString();
            out.write(">" + id + "\n");
            out.write(seq + "\n");
            seqFound = true;
        }
        out.close();
        if (seqFound == false)
        {
            System.out.println("Could not find a sequence with that name");
        }
    }

    public static void main(String[] args) {
       System.out.println("Please enter the filename: ");
       String outfile = scanner.nextLine();
       // File outfile = new File(out);
       FastaSubstrings test = new FastaSubstrings();

        try {
            InputStream fastain = new FileInputStream("P02768.fasta");
            test.getSequence1(fastain, outfile);
        } catch (FileNotFoundException ex) {
            System.out.println("Error reading file");
        }
        catch(IOException ex) {
            System.out.println("Error reading file");
        }
    }
}
