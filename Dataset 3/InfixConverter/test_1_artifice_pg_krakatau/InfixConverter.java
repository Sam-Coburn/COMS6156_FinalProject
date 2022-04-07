import java.util.Stack;

// 
// Decompiled by Procyon v0.5.29
// 

class InfixConverter
{
    private static String a(final String s) {
        final Stack<Character> stack = new Stack<Character>();
        String s2 = "";
        int n5;
        for (int i = 0; i < s.length(); i = n5) {
            final char char1 = s.charAt(i);
            String string = null;
            int n = 0;
            Label_0369: {
                if (char1 != '+') {
                    if (char1 != '-') {
                        if (char1 != '*') {
                            if (char1 != '/') {
                                string = s2 + char1;
                                n = i;
                                break Label_0369;
                            }
                        }
                    }
                }
                if (stack.size() > 0) {
                    final Character c = stack.peek();
                    int n2 = 0;
                    Label_0164: {
                        if (c != '*') {
                            if (c != '/') {
                                n2 = 0;
                                break Label_0164;
                            }
                        }
                        n2 = 1;
                    }
                    Label_0349: {
                        if (n2 != 1) {
                            if (char1 != '+') {
                                if (char1 != '-') {
                                    stack.push(char1);
                                    string = s2;
                                    n = i;
                                    break Label_0369;
                                }
                            }
                            final String string2 = s2 + stack.pop();
                            stack.push(char1);
                            string = string2;
                            n = i;
                        }
                        else {
                            if (char1 != '+') {
                                if (char1 != '-') {
                                    final String string3 = s2 + stack.pop();
                                    final int n3 = i - 1;
                                    string = string3;
                                    n = n3;
                                    break Label_0349;
                                }
                            }
                            final String string4 = s2 + stack.pop();
                            final int n4 = i - 1;
                            string = string4;
                            n = n4;
                        }
                    }
                }
                else {
                    stack.push(char1);
                    string = s2;
                    n = i;
                }
            }
            n5 = n + 1;
            s2 = string;
        }
        final int size = stack.size();
        int j = 0;
        String str = s2;
        while (j < size) {
            final String string5 = str + stack.pop();
            ++j;
            str = string5;
        }
        return str;
    }
    
    public static void main(final String[] array) {
        if (array.length != 1) {
            final String a = a("a+b*c");
            System.out.println("InFix  :\t" + "a+b*c");
            System.out.println("PostFix:\t" + a);
            System.out.println();
            final String a2 = a("a+b*c/d-e");
            System.out.println("InFix  :\t" + "a+b*c/d-e");
            System.out.println("PostFix:\t" + a2);
            System.out.println();
            final String a3 = a("a+b*c/d-e+f*h/i+j-k");
            System.out.println("InFix  :\t" + "a+b*c/d-e+f*h/i+j-k");
            System.out.println("PostFix:\t" + a3);
            System.out.println();
            return;
        }
        final String str = array[0];
        final String a4 = a(str);
        System.out.println("InFix  :\t" + str);
        System.out.println("PostFix:\t" + a4);
        System.out.println();
    }
}
