class InfixConverter {
    InfixConverter()
    {
        super();
    }
    
    public static String ReadString()
    {
        String s = null;
        try
        {
            s = new java.io.BufferedReader((java.io.Reader)new java.io.InputStreamReader(System.in)).readLine();
        }
        catch(Exception a)
        {
            a.printStackTrace();
            return "";
        }
        return s;
    }
    
    public static int ReadInteger()
    {
        int i = 0;
        try
        {
            i = Integer.parseInt(new java.io.BufferedReader((java.io.Reader)new java.io.InputStreamReader(System.in)).readLine());
        }
        catch(Exception a)
        {
            a.printStackTrace();
            return 0;
        }
        return i;
    }
    
    public static String InfixToPostfixConvert(String s)
    {
        java.util.Stack a = new java.util.Stack();
        String s0 = "";
        int i = 0;
        while(i < s.length())
        {
            int i0 = s.charAt(i);
            label3: {
                int i1 = 0;
                label12: {
                    label15: {
                        if (i0 != 43)
                        {
                            break label15;
                        }
                        i1 = 43;
                        break label12;
                    }
                    label14: {
                        if (i0 != 45)
                        {
                            break label14;
                        }
                        i1 = 45;
                        break label12;
                    }
                    label13: {
                        if (i0 != 42)
                        {
                            break label13;
                        }
                        i1 = 42;
                        break label12;
                    }
                    label11: {
                        if (i0 != 47)
                        {
                            break label11;
                        }
                        i1 = 47;
                        break label12;
                    }
                    StringBuilder a0 = new StringBuilder().append(s0);
                    int i2 = (char)i0;
                    s0 = a0.append((char)i2).toString();
                    break label3;
                }
                if (a.size() <= 0)
                {
                    int i3 = (char)i1;
                    a.push((Object)Character.valueOf((char)i3));
                }
                else
                {
                    boolean b = false;
                    Character a1 = (Character)a.peek();
                    int i4 = a1.charValue();
                    label10: {
                        label7: {
                            label9: {
                                label8: {
                                    if (i4 != 42)
                                    {
                                        break label8;
                                    }
                                    break label9;
                                }
                                int i5 = a1.charValue();
                                if (i5 != 47)
                                {
                                    break label7;
                                }
                            }
                            b = true;
                            break label10;
                        }
                        b = false;
                    }
                    if (b)
                    {
                        label4: {
                            label6: {
                                label5: {
                                    if (i1 != 43)
                                    {
                                        break label5;
                                    }
                                    break label6;
                                }
                                if (i1 != 45)
                                {
                                    break label4;
                                }
                            }
                            s0 = new StringBuilder().append(s0).append(a.pop()).toString();
                            i = i + -1;
                            break label3;
                        }
                        s0 = new StringBuilder().append(s0).append(a.pop()).toString();
                        i = i + -1;
                    }
                    else
                    {
                        int i6 = 0;
                        label1: {
                            label2: {
                                if (i1 != 43)
                                {
                                    break label2;
                                }
                                i6 = 43;
                                break label1;
                            }
                            label0: {
                                if (i1 != 45)
                                {
                                    break label0;
                                }
                                i6 = 45;
                                break label1;
                            }
                            int i7 = (char)i1;
                            a.push((Object)Character.valueOf((char)i7));
                            break label3;
                        }
                        s0 = new StringBuilder().append(s0).append(a.pop()).toString();
                        int i8 = (char)i6;
                        a.push((Object)Character.valueOf((char)i8));
                    }
                }
            }
            i = i + 1;
        }
        int i9 = a.size();
        int i10 = 0;
        while(i10 < i9)
        {
            s0 = new StringBuilder().append(s0).append(a.pop()).toString();
            i10 = i10 + 1;
        }
        return s0;
    }
    
    public static void main(String[] a)
    {
        if (a.length == 1)
        {
            String s = a[0];
            String s0 = InfixConverter.InfixToPostfixConvert(s);
            System.out.println(new StringBuilder().append("InFix  :\t").append(s).toString());
            System.out.println(new StringBuilder().append("PostFix:\t").append(s0).toString());
            System.out.println();
        }
        else
        {
            String s1 = InfixConverter.InfixToPostfixConvert("a+b*c");
            System.out.println(new StringBuilder().append("InFix  :\t").append("a+b*c").toString());
            System.out.println(new StringBuilder().append("PostFix:\t").append(s1).toString());
            System.out.println();
            String s2 = InfixConverter.InfixToPostfixConvert("a+b*c/d-e");
            System.out.println(new StringBuilder().append("InFix  :\t").append("a+b*c/d-e").toString());
            System.out.println(new StringBuilder().append("PostFix:\t").append(s2).toString());
            System.out.println();
            String s3 = InfixConverter.InfixToPostfixConvert("a+b*c/d-e+f*h/i+j-k");
            System.out.println(new StringBuilder().append("InFix  :\t").append("a+b*c/d-e+f*h/i+j-k").toString());
            System.out.println(new StringBuilder().append("PostFix:\t").append(s3).toString());
            System.out.println();
        }
    }
}
