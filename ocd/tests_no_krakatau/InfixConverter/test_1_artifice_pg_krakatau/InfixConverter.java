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
            label5: {
                int i1 = 0;
                label11: {
                    label14: {
                        if (i0 != 43)
                        {
                            break label14;
                        }
                        i1 = 43;
                        break label11;
                    }
                    label13: {
                        if (i0 != 45)
                        {
                            break label13;
                        }
                        i1 = 45;
                        break label11;
                    }
                    label12: {
                        if (i0 != 42)
                        {
                            break label12;
                        }
                        i1 = 42;
                        break label11;
                    }
                    label10: {
                        if (i0 != 47)
                        {
                            break label10;
                        }
                        i1 = 47;
                        break label11;
                    }
                    StringBuilder a0 = new StringBuilder().append(s0);
                    int i2 = (char)i0;
                    s0 = a0.append((char)i2).toString();
                    break label5;
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
                    label9: {
                        label6: {
                            label8: {
                                label7: {
                                    if (i4 != 42)
                                    {
                                        break label7;
                                    }
                                    break label8;
                                }
                                int i5 = a1.charValue();
                                if (i5 != 47)
                                {
                                    break label6;
                                }
                            }
                            b = true;
                            break label9;
                        }
                        b = false;
                    }
                    label3: {
                        label0: {
                            {
                                {
                                    int i6 = 0;
                                    label4: {
                                        if (!b)
                                        {
                                            break label4;
                                        }
                                        if (i1 == 43)
                                        {
                                            break label3;
                                        }
                                        if (i1 == 45)
                                        {
                                            break label3;
                                        }
                                        s0 = new StringBuilder().append(s0).append(a.pop()).toString();
                                        i = i + -1;
                                        break label5;
                                    }
                                    label2: {
                                        label1: {
                                            if (i1 != 43)
                                            {
                                                break label1;
                                            }
                                            i6 = 43;
                                            break label2;
                                        }
                                        if (i1 != 45)
                                        {
                                            break label0;
                                        }
                                        i6 = 45;
                                    }
                                    s0 = new StringBuilder().append(s0).append(a.pop()).toString();
                                    int i7 = (char)i6;
                                    a.push((Object)Character.valueOf((char)i7));
                                }
                                break label5;
                            }
                        }
                        int i8 = (char)i1;
                        a.push((Object)Character.valueOf((char)i8));
                        break label5;
                    }
                    s0 = new StringBuilder().append(s0).append(a.pop()).toString();
                    i = i + -1;
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
            String s0 = InfixConverter.a(s);
            System.out.println(new StringBuilder("InFix  :\t").append(s).toString());
            System.out.println(new StringBuilder("PostFix:\t").append(s0).toString());
            System.out.println();
            return;
        }
        String s1 = InfixConverter.a("a+b*c");
        System.out.println(new StringBuilder("InFix  :\t").append("a+b*c").toString());
        System.out.println(new StringBuilder("PostFix:\t").append(s1).toString());
        System.out.println();
        String s2 = InfixConverter.a("a+b*c/d-e");
        System.out.println(new StringBuilder("InFix  :\t").append("a+b*c/d-e").toString());
        System.out.println(new StringBuilder("PostFix:\t").append(s2).toString());
        System.out.println();
        String s3 = InfixConverter.a("a+b*c/d-e+f*h/i+j-k");
        System.out.println(new StringBuilder("InFix  :\t").append("a+b*c/d-e+f*h/i+j-k").toString());
        System.out.println(new StringBuilder("PostFix:\t").append(s3).toString());
        System.out.println();
    }
}
