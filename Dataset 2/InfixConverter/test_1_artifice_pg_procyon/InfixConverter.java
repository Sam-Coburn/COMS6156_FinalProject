class InfixConverter {
    InfixConverter()
    {
        super();
    }
    
    private static String a(String s)
    {
        java.util.Stack a = new java.util.Stack();
        String s0 = "";
        int i = 0;
        while(i < s.length())
        {
            int i0 = s.charAt(i);
            label2: {
                label5: {
                    if (i0 == 43)
                    {
                        break label5;
                    }
                    if (i0 == 45)
                    {
                        break label5;
                    }
                    if (i0 == 42)
                    {
                        break label5;
                    }
                    if (i0 == 47)
                    {
                        break label5;
                    }
                    s0 = new StringBuilder().append(s0).append((char)i0).toString();
                    break label2;
                }
                if (a.size() > 0)
                {
                    Character a0 = (Character)a.peek();
                    int i1 = a0.charValue();
                    {
                        label4: {
                            if (i1 == 42)
                            {
                                break label4;
                            }
                            int i2 = a0.charValue();
                            if (i2 == 47)
                            {
                                break label4;
                            }
                            label3: {
                                if (i0 == 43)
                                {
                                    break label3;
                                }
                                if (i0 == 45)
                                {
                                    break label3;
                                }
                                a.push((Object)Character.valueOf((char)i0));
                                break label2;
                            }
                            s0 = new StringBuilder().append(s0).append(a.pop()).toString();
                            a.push((Object)Character.valueOf((char)i0));
                            break label2;
                        }
                        label0: {
                            label1: {
                                if (i0 == 43)
                                {
                                    break label1;
                                }
                                if (i0 != 45)
                                {
                                    break label0;
                                }
                            }
                            s0 = new StringBuilder().append(s0).append(a.pop()).toString();
                            i = i + -1;
                            break label2;
                        }
                        s0 = new StringBuilder().append(s0).append(a.pop()).toString();
                        i = i + -1;
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
        if (a.length == 1)
        {
            String s = a[0];
            String s0 = InfixConverter.a(s);
            System.out.println(new StringBuilder().append("InFix  :\t").append(s).toString());
            System.out.println(new StringBuilder().append("PostFix:\t").append(s0).toString());
            System.out.println();
            return;
        }
        String s1 = InfixConverter.a("a+b*c");
        System.out.println(new StringBuilder().append("InFix  :\t").append("a+b*c").toString());
        System.out.println(new StringBuilder().append("PostFix:\t").append(s1).toString());
        System.out.println();
        String s2 = InfixConverter.a("a+b*c/d-e");
        System.out.println(new StringBuilder().append("InFix  :\t").append("a+b*c/d-e").toString());
        System.out.println(new StringBuilder().append("PostFix:\t").append(s2).toString());
        System.out.println();
        String s3 = InfixConverter.a("a+b*c/d-e+f*h/i+j-k");
        System.out.println(new StringBuilder().append("InFix  :\t").append("a+b*c/d-e+f*h/i+j-k").toString());
        System.out.println(new StringBuilder().append("PostFix:\t").append(s3).toString());
        System.out.println();
    }
}
