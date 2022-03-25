class SqrtAlgorithm {
    SqrtAlgorithm()
    {
        super();
    }
    
    public static double SqrtByAlogorithm(double d)
    {
        int i = 0;
        boolean b = false;
        int i0 = 0;
        RuntimeException a = null;
        long j = (long)d;
        long j0 = (long)((d - (double)j) * 1000000.0);
        long j1 = j;
        int i1 = 0;
        while(j1 >= 10L)
        {
            j1 = j1 / 10L;
            i1 = i1 + 1;
        }
        int i2 = i1 + 1;
        long j2 = j0;
        int i3 = 0;
        while(j2 >= 10L)
        {
            j2 = j2 / 10L;
            i3 = i3 + 1;
        }
        int i4 = i3 + 1;
        if (i2 % 2 == 1)
        {
            i = 0;
            b = true;
            i0 = 0;
        }
        else
        {
            i = 0;
            b = false;
            i0 = 0;
        }
        try
        {
            while(true)
            {
                int i5 = 0;
                boolean b0 = false;
                if (b)
                {
                    i5 = i2 - 1;
                    b0 = true;
                }
                else
                {
                    i5 = i2 - 2;
                    b0 = false;
                }
                int i6 = 1;
                int i7 = 0;
                while(i7 < i5)
                {
                    i6 = i6 * 10;
                    i7 = i7 + 1;
                }
                int i8 = (int)j / i6;
                int i9 = i * 2;
                int i10 = 1;
                while(true)
                {
                    double d0 = 0.0;
                    label12: {
                        label10: {
                            label9: {
                                label11: {
                                    if (i9 != 0)
                                    {
                                        break label11;
                                    }
                                    if (i8 - i10 * i10 < 0)
                                    {
                                        break label10;
                                    }
                                    i9 = 0;
                                    break label9;
                                }
                                if (i8 - i10 * (i9 * 10 + i10) >= 0)
                                {
                                    break label9;
                                }
                                i = i9 / 2 * 10 + i10 - 1;
                                break label12;
                            }
                            i10 = i10 + 1;
                            continue;
                        }
                        i = i10 - 1;
                    }
                    int i11 = i / 10;
                    int i12 = i % 10;
                    j = (i11 == 0) ? j - (long)(i * i * i6) : j - (long)((i11 * 2 * 10 + i12) * i12 * i6);
                    int i13 = (j >= 0L) ? (j != 0L) ? 1 : 0 : -1;
                    label1: {
                        label7: {
                            label8: {
                                if (i13 == 0)
                                {
                                    break label8;
                                }
                                break label7;
                            }
                            int i14 = (j0 < 0L) ? -1 : (j0 == 0L) ? 0 : 1;
                            label6: {
                                if (i14 == 0)
                                {
                                    break label6;
                                }
                                break label7;
                            }
                            if (i5 <= 0)
                            {
                                break label1;
                            }
                            int i15 = i5 / 2;
                            int i16 = 1;
                            int i17 = 0;
                            while(i17 < i15)
                            {
                                i16 = i16 * 10;
                                i17 = i17 + 1;
                            }
                            i = i * i16;
                            break label1;
                        }
                        i2 = b0 ? i2 + -1 : i2 + -2;
                        label5: {
                            label4: {
                                if (i2 <= 0)
                                {
                                    break label4;
                                }
                                break label5;
                            }
                            int i18 = (j >= 0L) ? (j != 0L) ? 1 : 0 : -1;
                            label3: {
                                label2: {
                                    if (i18 <= 0)
                                    {
                                        break label2;
                                    }
                                    break label3;
                                }
                                if (j0 <= 0L)
                                {
                                    break label1;
                                }
                            }
                            label0: {
                                if (i0 < 5)
                                {
                                    break label0;
                                }
                                break label1;
                            }
                            i0 = i0 + 1;
                            j = j * 100L;
                            if (j0 > 0L)
                            {
                                i4 = i4 + -2;
                                int i19 = 1;
                                int i20 = 0;
                                while(i20 < i4)
                                {
                                    i19 = i19 * 10;
                                    i20 = i20 + 1;
                                }
                                j = j + j0 / (long)i19;
                                j0 = j0 % (long)i19;
                            }
                            i2 = i2 + 2;
                        }
                        b = false;
                        break;
                    }
                    if (i0 == 0)
                    {
                        d0 = (double)i;
                    }
                    else
                    {
                        int i21 = 1;
                        int i22 = 0;
                        while(i22 < i0)
                        {
                            i21 = i21 * 10;
                            i22 = i22 + 1;
                        }
                        d0 = (double)i / (double)i21;
                    }
                    return d0;
                }
            }
        }
        catch(ArithmeticException | IllegalMonitorStateException a0)
        {
            a = a0;
        }
        throw a;
    }
    
    public static void main(String[] a)
    {
        double d = 0.0;
        while(d <= 10000.0)
        {
            System.out.print("sqrt(");
            System.out.print(d);
            System.out.print(") = ");
            System.out.print(SqrtAlgorithm.SqrtByAlogorithm(d));
            System.out.print(", ");
            System.out.println(Math.sqrt(d));
            d = d + 50.0;
        }
    }
}
