class SqrtAlgorithm {
    SqrtAlgorithm()
    {
        super();
    }
    
    public static void main(String[] a)
    {
        double d = 0.0;
        label12: while(d <= 10000.0)
        {
            int i = 0;
            boolean b = false;
            int i0 = 0;
            System.out.print("sqrt(");
            System.out.print(d);
            System.out.print(") = ");
            java.io.PrintStream a0 = System.out;
            long j = (long)d;
            long j0 = (long)((d - (double)j) * 1000000.0);
            int i1 = 0;
            long j1 = j;
            while(j1 >= 10L)
            {
                j1 = j1 / 10L;
                i1 = i1 + 1;
            }
            int i2 = i1 + 1;
            int i3 = 0;
            long j2 = j0;
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
                int i6 = 0;
                int i7 = 1;
                while(i6 < i5)
                {
                    i7 = i7 * 10;
                    i6 = i6 + 1;
                }
                int i8 = (int)j / i7;
                int i9 = i << 1;
                int i10 = 1;
                while(true)
                {
                    double d0 = 0.0;
                    label11: {
                        label9: {
                            label8: {
                                label10: {
                                    if (i9 != 0)
                                    {
                                        break label10;
                                    }
                                    if (i8 - i10 * i10 < 0)
                                    {
                                        break label9;
                                    }
                                    i9 = 0;
                                    break label8;
                                }
                                if (i8 - i10 * (i9 * 10 + i10) >= 0)
                                {
                                    break label8;
                                }
                                i = i9 / 2 * 10 + i10 - 1;
                                break label11;
                            }
                            i10 = i10 + 1;
                            continue;
                        }
                        i = i10 - 1;
                    }
                    int i11 = i / 10;
                    int i12 = i % 10;
                    j = (i11 == 0) ? j - (long)(-(i * i * i7)) : j - (long)(-(((i11 << 1) * 10 + i12) * i12 * i7));
                    int i13 = (j >= 0L) ? (j != 0L) ? 1 : 0 : -1;
                    label0: {
                        label6: {
                            label7: {
                                if (i13 == 0)
                                {
                                    break label7;
                                }
                                break label6;
                            }
                            int i14 = (j0 < 0L) ? -1 : (j0 == 0L) ? 0 : 1;
                            label5: {
                                if (i14 == 0)
                                {
                                    break label5;
                                }
                                break label6;
                            }
                            if (i5 <= 0)
                            {
                                break label0;
                            }
                            int i15 = i5 / 2;
                            int i16 = 0;
                            int i17 = 1;
                            while(i16 < i15)
                            {
                                i17 = i17 * 10;
                                i16 = i16 + 1;
                            }
                            i = i * i17;
                            break label0;
                        }
                        i2 = b0 ? i2 + -1 : i2 + -2;
                        label4: {
                            label3: {
                                if (i2 <= 0)
                                {
                                    break label3;
                                }
                                break label4;
                            }
                            int i18 = (j >= 0L) ? (j != 0L) ? 1 : 0 : -1;
                            label2: {
                                label1: {
                                    if (i18 <= 0)
                                    {
                                        break label1;
                                    }
                                    break label2;
                                }
                                if (j0 <= 0L)
                                {
                                    break label0;
                                }
                            }
                            if (i0 >= 5)
                            {
                                break label0;
                            }
                            i0 = i0 + 1;
                            j = j * 100L;
                            if (j0 > 0L)
                            {
                                i4 = i4 + -2;
                                int i19 = 0;
                                int i20 = 1;
                                while(i19 < i4)
                                {
                                    i20 = i20 * 10;
                                    i19 = i19 + 1;
                                }
                                j = j + j0 / (long)i20;
                                j0 = j0 % (long)i20;
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
                        int i21 = 0;
                        int i22 = 1;
                        while(i21 < i0)
                        {
                            i22 = i22 * 10;
                            i21 = i21 + 1;
                        }
                        d0 = (double)i / (double)i22;
                    }
                    a0.print(d0);
                    System.out.print(", ");
                    System.out.println(Math.sqrt(d));
                    d = d + 50.0;
                    continue label12;
                }
            }
        }
    }
}
