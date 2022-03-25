import java.util.Stack;
import java.io.Reader;
import java.io.BufferedReader;
import java.io.InputStreamReader;

// 
// Decompiled by Procyon v0.5.29
// 

class InfixConverter
{
    public static String ReadString() {
        try {
            return new BufferedReader(new InputStreamReader(System.in)).readLine();
        }
        catch (Exception ex) {
            ex.printStackTrace();
            return "";
        }
    }
    
    public static int ReadInteger() {
        try {
            return Integer.parseInt(new BufferedReader(new InputStreamReader(System.in)).readLine());
        }
        catch (Exception ex) {
            ex.printStackTrace();
            return 0;
        }
    }
    
    public static String InfixToPostfixConvert(final String s) {
        String str = "";
        final Stack<Character> stack = new Stack<Character>();
        for (int i = 0; i < s.length(); ++i) {
            final char char1 = s.charAt(i);
            if (char1 == '+' || char1 == '-' || char1 == '*' || char1 == '/') {
                if (stack.size() <= 0) {
                    stack.push(char1);
                }
                else {
                    final Character c = stack.peek();
                    if (c == '*' || c == '/') {
                        if (char1 == '+' || char1 == '-') {
                            str += stack.pop();
                            --i;
                        }
                        else {
                            str += stack.pop();
                            --i;
                        }
                    }
                    else if (char1 == '+' || char1 == '-') {
                        str += stack.pop();
                        stack.push(char1);
                    }
                    else {
                        stack.push(char1);
                    }
                }
            }
            else {
                str += char1;
            }
        }
        for (int size = stack.size(), j = 0; j < size; ++j) {
            str += stack.pop();
        }
        return str;
    }
    
    public static void main(final String[] array) {
        if (array.length == 1) {
            final String str = array[0];
            final String infixToPostfixConvert = InfixToPostfixConvert(str);
            System.out.println("InFix  :\t" + str);
            System.out.println("PostFix:\t" + infixToPostfixConvert);
            System.out.println();
        }
        else {
            final String str2 = "a+b*c";
            final String infixToPostfixConvert2 = InfixToPostfixConvert(str2);
            System.out.println("InFix  :\t" + str2);
            System.out.println("PostFix:\t" + infixToPostfixConvert2);
            System.out.println();
            final String str3 = "a+b*c/d-e";
            final String infixToPostfixConvert3 = InfixToPostfixConvert(str3);
            System.out.println("InFix  :\t" + str3);
            System.out.println("PostFix:\t" + infixToPostfixConvert3);
            System.out.println();
            final String str4 = "a+b*c/d-e+f*h/i+j-k";
            final String infixToPostfixConvert4 = InfixToPostfixConvert(str4);
            System.out.println("InFix  :\t" + str4);
            System.out.println("PostFix:\t" + infixToPostfixConvert4);
            System.out.println();
        }
    }
}
