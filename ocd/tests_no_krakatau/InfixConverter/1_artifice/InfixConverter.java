class InfixConverter {
    InfixConverter()
    {
        super();
    }
    
    public static String m00()
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
    
    public static int m10()
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
    
    public static String m20(String s)
    {
        java.util.Stack a = new java.util.Stack();
        String s0 = "";
        int i = 0;
        while(i < s.length())
        {
            int i0 = s.charAt(i);
            label1: {
                label7: {
                    if (i0 == 43)
                    {
                        break label7;
                    }
                    if (i0 == 45)
                    {
                        break label7;
                    }
                    if (i0 == 42)
                    {
                        break label7;
                    }
                    if (i0 == 47)
                    {
                        break label7;
                    }
                    s0 = new StringBuilder().append(s0).append((char)i0).toString();
                    break label1;
                }
                if (a.size() > 0)
                {
                    boolean b = false;
                    Character a0 = (Character)a.peek();
                    int i1 = a0.charValue();
                    label6: {
                        label4: {
                            label5: {
                                if (i1 == 42)
                                {
                                    break label5;
                                }
                                int i2 = a0.charValue();
                                if (i2 != 47)
                                {
                                    break label4;
                                }
                            }
                            b = true;
                            break label6;
                        }
                        b = false;
                    }
                    if (b)
                    {
                        label2: {
                            label3: {
                                if (i0 == 43)
                                {
                                    break label3;
                                }
                                if (i0 != 45)
                                {
                                    break label2;
                                }
                            }
                            s0 = new StringBuilder().append(s0).append(a.pop()).toString();
                            i = i - 1;
                            break label1;
                        }
                        s0 = new StringBuilder().append(s0).append(a.pop()).toString();
                        i = i - 1;
                    }
                    else
                    {
                        label0: {
                            if (i0 == 43)
                            {
                                break label0;
                            }
                            if (i0 == 45)
                            {
                                break label0;
                            }
                            a.push((Object)Character.valueOf((char)i0));
                            break label1;
                        }
                        s0 = new StringBuilder().append(s0).append(a.pop()).toString();
                        a.push((Object)Character.valueOf((char)i0));
                    }
                }
                else
                {
                    a.push((Object)Character.valueOf((char)i0));
                }
            }
            i = i + 1;
        }
        int i3 = a.size();
        int i4 = 0;
        while(i4 < i3)
        {
            s0 = new StringBuilder().append(s0).append(a.pop()).toString();
            i4 = i4 + 1;
        }
        return s0;
    }
    
    public static void main(String[] a)
    {
        if (a.length != 1)
        {
            String s = InfixConverter.m20("a+b*c");
            System.out.println(new StringBuilder().append("InFix  :\t").append("a+b*c").toString());
            System.out.println(new StringBuilder().append("PostFix:\t").append(s).toString());
            System.out.println();
            String s0 = InfixConverter.m20("a+b*c/d-e");
            System.out.println(new StringBuilder().append("InFix  :\t").append("a+b*c/d-e").toString());
            System.out.println(new StringBuilder().append("PostFix:\t").append(s0).toString());
            System.out.println();
            String s1 = InfixConverter.m20("a+b*c/d-e+f*h/i+j-k");
            System.out.println(new StringBuilder().append("InFix  :\t").append("a+b*c/d-e+f*h/i+j-k").toString());
            System.out.println(new StringBuilder().append("PostFix:\t").append(s1).toString());
            System.out.println();
        }
        else
        {
            String s2 = a[0];
            String s3 = InfixConverter.m20(s2);
            System.out.println(new StringBuilder().append("InFix  :\t").append(s2).toString());
            System.out.println(new StringBuilder().append("PostFix:\t").append(s3).toString());
            System.out.println();
        }
    }
}
