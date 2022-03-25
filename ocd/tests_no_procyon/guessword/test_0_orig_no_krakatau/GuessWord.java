import java.io.InputStreamReader;
import java.io.IOException;
import java.io.FileNotFoundException;
import java.io.Reader;
import java.io.BufferedReader;
import java.io.FileReader;

// 
// Decompiled by Procyon v0.5.29
// 

class GuessWord
{
    public static int ReadWordsFromFile(final String[] array) {
        FileNotFoundException ex;
        try {
            try {
                final FileReader in = new FileReader("words_input.txt");
                final BufferedReader bufferedReader = new BufferedReader(in);
                int n = 0;
                int n2;
                for (int i = 0; i < 100; ++i, n = n2) {
                    final String line = bufferedReader.readLine();
                    if (line == null) {
                        break;
                    }
                    n2 = n + 1;
                    array[n] = line;
                }
                in.close();
                return n;
            }
            catch (FileNotFoundException ex2) {
                ex = ex2;
            }
        }
        catch (IOException ex3) {
            System.out.println(ex3.getStackTrace());
            return -1;
        }
        System.out.println(ex.getMessage());
        return -1;
    }
    
    public static String ReadString() {
        String line;
        try {
            line = new BufferedReader(new InputStreamReader(System.in)).readLine();
        }
        catch (Exception ex) {
            ex.printStackTrace();
            return "";
        }
        return line;
    }
    
    public static void main(final String[] array) {
        System.out.println("Welcome to Guess a Word\n");
        final String[] array2 = new String[100];
        final int readWordsFromFile = ReadWordsFromFile(array2);
        if (readWordsFromFile < 0) {
            System.out.println("No words found in the file");
            return;
        }
        final String str = array2[(int)(Math.random() * 100.0) % readWordsFromFile];
        final int length = str.length();
        System.out.print("Your secret word is: ");
        for (int i = 0; i < length; ++i) {
            System.out.print("*");
        }
        System.out.println();
        System.out.println("Guess now  (To stop the program, enter #) : ");
        String str2 = "";
        boolean b;
        while (true) {
            final String readString = ReadString();
            if (readString.compareTo("#") == 0) {
                b = false;
                break;
            }
            str2 += readString;
            if (str2.length() > 10) {
                str2 = str2.substring(0, 9);
            }
            boolean b2 = true;
            boolean b3 = false;
            String s = "";
            int j = 0;
        Label_0177:
            while (j < length) {
                while (true) {
                    for (int k = 0; k < str2.length(); ++k) {
                        if (readString.indexOf(str.charAt(j)) >= 0) {
                            b3 = true;
                        }
                        if (str.charAt(j) == str2.charAt(k)) {
                            s += str.charAt(j);
                            final boolean b4 = true;
                            if (!b4) {
                                s += "*";
                                b2 = false;
                            }
                            ++j;
                            continue Label_0177;
                        }
                    }
                    final boolean b4 = false;
                    continue;
                }
            }
            if (b3) {
                System.out.format("Correct: %s", s);
            }
            else {
                System.out.format("Wrong: %s", s);
            }
            if (b2) {
                b = true;
                break;
            }
            System.out.println();
            System.out.print("Guesses : ");
            for (int l = 0; l < str2.length(); ++l) {
                System.out.format("%c ", str2.charAt(l));
            }
            System.out.println();
            System.out.println();
            if (str2.length() < 10) {
                continue;
            }
            System.out.print("You have reached maximum amount of guesses.");
            b = false;
            break;
        }
        if (b) {
            System.out.println("\nCongrats! You have guessed it correctly");
        }
        else {
            System.out.println("\nUnfortunately you did not guess it correctly. The secret word is: " + str);
        }
    }
}
