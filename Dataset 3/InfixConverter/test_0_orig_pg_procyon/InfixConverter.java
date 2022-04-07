import java.util.Stack;

// 
// Decompiled by Procyon v0.5.29
// 

class InfixConverter
{
    private static String a(final String s) {
        String str = "";
        final Stack<Character> stack = new Stack<Character>();
        for (int i = 0; i < s.length(); ++i) {
            final char char1;
            if ((char1 = s.charAt(i)) == '+' || char1 == '-' || char1 == '*' || char1 == '/') {
                if (stack.size() <= 0) {
                    stack.push(char1);
                }
                else {
                    final Character c;
                    if ((c = stack.peek()) == '*' || c == '/') {
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
            final String str;
            final String a = a(str = array[0]);
            System.out.println("InFix  :\t" + str);
            System.out.println("PostFix:\t" + a);
            System.out.println();
            return;
        }
        final String str2;
        final String a2 = a(str2 = "a+b*c");
        System.out.println("InFix  :\t" + str2);
        System.out.println("PostFix:\t" + a2);
        System.out.println();
        final String str3;
        final String a3 = a(str3 = "a+b*c/d-e");
        System.out.println("InFix  :\t" + str3);
        System.out.println("PostFix:\t" + a3);
        System.out.println();
        final String str4;
        final String a4 = a(str4 = "a+b*c/d-e+f*h/i+j-k");
        System.out.println("InFix  :\t" + str4);
        System.out.println("PostFix:\t" + a4);
        System.out.println();
    }
}
