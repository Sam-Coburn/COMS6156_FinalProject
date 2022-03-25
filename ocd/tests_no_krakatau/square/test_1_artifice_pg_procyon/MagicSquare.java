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
        System.out.println(new StringBuilder().append("There were ").append(a1.e).append(" number of solutions.").toString());
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
        while(true)
        {
            int i1 = 0;
            while(i1 < this.a.length)
            {
                int i2 = 0;
                boolean b0 = false;
                int i3 = 0;
                while(i3 < this.a[i1].length)
                {
                    i2 = i2 + this.a[i1][i3];
                    if (this.a[i1][i3] == 0)
                    {
                        b0 = true;
                    }
                    i3 = i3 + 1;
                }
                if (!b0 && i2 != this.d)
                {
                    return;
                }
                i1 = i1 + 1;
            }
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
                s = new StringBuilder().append(s).append(0 + this.a[i][i0]).append("\t").toString();
                i0 = i0 + 1;
            }
            s = new StringBuilder().append(s).append("0\n").toString();
            i = i + 1;
        }
        return s;
    }
}
