import java.util.Stack;
import java.io.Reader;
import java.io.BufferedReader;
import java.io.InputStreamReader;

// 
// Decompiled by Procyon v0.5.29
// 

class InfixConverter
{
    public static String m00() {
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
    
    public static int m10() {
        int int1;
        try {
            int1 = Integer.parseInt(new BufferedReader(new InputStreamReader(System.in)).readLine());
        }
        catch (Exception ex) {
            ex.printStackTrace();
            return 0;
        }
        return int1;
    }
    
    public static String m20(final String s) {
        final Stack<Character> stack = new Stack<Character>();
        String s2 = "";
        int n6;
        for (int i = 0; i < s.length(); i = n6) {
            final char char1 = s.charAt(i);
            String string = null;
            int n = 0;
            Label_0383: {
                if (char1 != '+') {
                    if (char1 != '-') {
                        if (char1 != '*') {
                            if (char1 != '/') {
                                string = s2 + char1;
                                n = i;
                                break Label_0383;
                            }
                        }
                    }
                }
                if (stack.size() > 0) {
                    final Character c = stack.peek();
                    int n2 = 0;
                    Label_0170: {
                        if (c != '*') {
                            if (c != '/') {
                                n2 = 0;
                                break Label_0170;
                            }
                        }
                        n2 = 1;
                    }
                    String s3 = null;
                    int n3 = 0;
                    Label_0355: {
                        if (n2 != 1) {
                            if (char1 != '+') {
                                if (char1 != '-') {
                                    stack.push(char1);
                                    s3 = s2;
                                    n3 = i;
                                    break Label_0355;
                                }
                            }
                            final String string2 = s2 + stack.pop();
                            stack.push(char1);
                            s3 = string2;
                            n3 = i;
                        }
                        else {
                            if (char1 != '+') {
                                if (char1 != '-') {
                                    final String string3 = s2 + stack.pop();
                                    final int n4 = i - 1;
                                    s3 = string3;
                                    n3 = n4;
                                    break Label_0355;
                                }
                            }
                            final String string4 = s2 + stack.pop();
                            final int n5 = i - 1;
                            s3 = string4;
                            n3 = n5;
                        }
                    }
                    string = s3;
                    n = n3;
                }
                else {
                    stack.push(char1);
                    string = s2;
                    n = i;
                }
            }
            n6 = n + 1;
            s2 = string;
        }
        final int size = stack.size();
        String str = s2;
        int n7;
        for (int j = 0; j < size; j = n7) {
            final String string5 = str + stack.pop();
            n7 = j + 1;
            str = string5;
        }
        return str;
    }
    
    public static void main(final String[] array) {
        if (array.length != 1) {
            final String m20 = m20("a+b*c");
            System.out.println("InFix  :\t" + "a+b*c");
            System.out.println("PostFix:\t" + m20);
            System.out.println();
            final String m2 = m20("a+b*c/d-e");
            System.out.println("InFix  :\t" + "a+b*c/d-e");
            System.out.println("PostFix:\t" + m2);
            System.out.println();
            final String m3 = m20("a+b*c/d-e+f*h/i+j-k");
            System.out.println("InFix  :\t" + "a+b*c/d-e+f*h/i+j-k");
            System.out.println("PostFix:\t" + m3);
            System.out.println();
        }
        else {
            final String str = array[0];
            final String m4 = m20(str);
            System.out.println("InFix  :\t" + str);
            System.out.println("PostFix:\t" + m4);
            System.out.println();
        }
    }
}
