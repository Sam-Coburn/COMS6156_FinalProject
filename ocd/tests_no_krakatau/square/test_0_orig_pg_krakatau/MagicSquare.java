public class MagicSquare {
    private int[][] a;
    private boolean[] b;
    private int c;
    private int d;
    private int e;
    
    public static void main(String[] a0)
    {
        MagicSquare a1 = new MagicSquare(3);
        a1.a(0, 0);
        System.out.println(new StringBuilder("There were ").append(a1.e).append(" number of solutions.").toString());
    }
    
    private MagicSquare(int i)
    {
        super();
        this.a = new int[3][3];
        int i0 = 0;
        while(i0 < 3)
        {
            int i1 = 0;
            while(i1 < 3)
            {
                this.a[i0][i1] = 0;
                i1 = i1 + 1;
            }
            i0 = i0 + 1;
        }
        this.c = 9;
        this.b = new boolean[this.c];
        int i2 = 0;
        while(i2 < this.c)
        {
            this.b[i2] = true;
            i2 = i2 + 1;
        }
        this.d = 15;
        this.e = 0;
    }
    
    private void a(int i, int i0)
    {
        int i1 = 0;
        while(true)
        {
            boolean b0 = false;
            if (i1 < this.a.length)
            {
                int i2 = 0;
                boolean b1 = false;
                int i3 = 0;
                while(i3 < this.a[i1].length)
                {
                    i2 = i2 + this.a[i1][i3];
                    if (!(this.a[i1][i3] != 0))
                    {
                        b1 = true;
                    }
                    i3 = i3 + 1;
                }
                label9: {
                    label11: {
                        label10: {
                            if (!b1)
                            {
                                break label10;
                            }
                            break label11;
                        }
                        if (i2 != this.d)
                        {
                            break label9;
                        }
                    }
                    i1 = i1 + 1;
                    continue;
                }
                b0 = false;
            }
            else
            {
                b0 = true;
            }
            if (b0)
            {
                boolean b2 = false;
                int i4 = 0;
                while(true)
                {
                    if (i4 < this.a[0].length)
                    {
                        int i5 = 0;
                        boolean b3 = false;
                        int i6 = 0;
                        while(i6 < this.a.length)
                        {
                            i5 = i5 + this.a[i6][i4];
                            if (!(this.a[i6][i4] != 0))
                            {
                                b3 = true;
                            }
                            i6 = i6 + 1;
                        }
                        label6: {
                            label8: {
                                label7: {
                                    if (!b3)
                                    {
                                        break label7;
                                    }
                                    break label8;
                                }
                                if (i5 != this.d)
                                {
                                    break label6;
                                }
                            }
                            i4 = i4 + 1;
                            continue;
                        }
                        b2 = false;
                        break;
                    }
                    else
                    {
                        b2 = true;
                        break;
                    }
                }
                if (b2)
                {
                    boolean b4 = false;
                    int i7 = 0;
                    boolean b5 = false;
                    int i8 = 0;
                    while(i8 < this.a.length)
                    {
                        i7 = i7 + this.a[i8][i8];
                        if (!(this.a[i8][i8] != 0))
                        {
                            b5 = true;
                        }
                        i8 = i8 + 1;
                    }
                    label3: {
                        label4: {
                            label5: {
                                if (!b5)
                                {
                                    break label5;
                                }
                                break label4;
                            }
                            if (i7 == this.d)
                            {
                                break label4;
                            }
                            b4 = false;
                            break label3;
                        }
                        int i9 = 0;
                        boolean b6 = false;
                        int i10 = 0;
                        while(i10 < this.a.length)
                        {
                            i9 = i9 + this.a[i10][this.a.length - 1 - i10];
                            if (!(this.a[i10][this.a.length - 1 - i10] != 0))
                            {
                                b6 = true;
                            }
                            i10 = i10 + 1;
                        }
                        label0: {
                            label2: {
                                label1: {
                                    if (!b6)
                                    {
                                        break label1;
                                    }
                                    break label2;
                                }
                                if (i9 != this.d)
                                {
                                    break label0;
                                }
                            }
                            b4 = true;
                            break label3;
                        }
                        b4 = false;
                    }
                    if (b4)
                    {
                        if (i == this.a.length)
                        {
                            System.out.println((Object)this);
                            this.e = this.e + 1;
                            return;
                        }
                        int i11 = 0;
                        while(i11 < this.c)
                        {
                            if (this.b[i11])
                            {
                                int i12 = 0;
                                this.a[i][i0] = i11 + 1;
                                this.b[i11] = false;
                                int i13 = i0 + 1;
                                if (i13 == this.a.length)
                                {
                                    i12 = i + 1;
                                    i13 = 0;
                                }
                                else
                                {
                                    i12 = i;
                                }
                                this.a(i12, i13);
                                this.a[i][i0] = 0;
                                this.b[i11] = true;
                            }
                            i11 = i11 + 1;
                        }
                        return;
                    }
                }
            }
            return;
        }
    }
    
    public String toString()
    {
        String s = "";
        int i = 0;
        while(i < this.a.length)
        {
            int i0 = 0;
            while(i0 < this.a[i].length)
            {
                s = new StringBuilder().append(s).append((this.a[i][i0] != 0) ? 1 : 0).append("\t").toString();
                i0 = i0 + 1;
            }
            s = new StringBuilder().append(s).append("\n").toString();
            i = i + 1;
        }
        return s;
    }
}
