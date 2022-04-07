public class MagicSquare {
    private int[][] f00;
    private boolean[] f10;
    private int f20;
    private int f30;
    private int f40;
    
    public static void main(String[] a)
    {
        MagicSquare a0 = new MagicSquare(3);
        a0.m00(0, 0);
        System.out.println(new StringBuilder().append("There were ").append(a0.m40()).append(" number of solutions.").toString());
    }
    
    public MagicSquare(int i)
    {
        super();
        this.f00 = new int[i][i];
        int i0 = 0;
        while(i0 < i)
        {
            int i1 = 0;
            while(i1 < i)
            {
                this.f00[i0][i1] = 0;
                i1 = i1 + 1;
            }
            i0 = i0 + 1;
        }
        this.f20 = i * i;
        this.f10 = new boolean[this.f20];
        int i2 = 0;
        while(i2 < this.f20)
        {
            this.f10[i2] = true;
            i2 = i2 + 1;
        }
        this.f30 = i * (i * i + 1) / 2;
        this.f40 = 0;
    }
    
    public void m00(int i, int i0)
    {
        if (this.m10() && this.m20() && this.m30())
        {
            if (i == this.f00.length)
            {
                System.out.println((Object)this);
                this.f40 = this.f40 + 1;
                return;
            }
            int i1 = 0;
            while(i1 < this.f20)
            {
                if (this.f10[i1])
                {
                    int i2 = 0;
                    this.f00[i][i0] = i1 + 1;
                    this.f10[i1] = false;
                    int i3 = i0 + 1;
                    if (i3 == this.f00.length)
                    {
                        i2 = i + 1;
                        i3 = 0;
                    }
                    else
                    {
                        i2 = i;
                    }
                    this.m00(i2, i3);
                    this.f00[i][i0] = 0;
                    this.f10[i1] = true;
                }
                i1 = i1 + 1;
            }
            return;
        }
    }
    
    public boolean m10()
    {
        int i = 0;
        while(i < this.f00.length)
        {
            int i0 = 0;
            boolean b = false;
            int i1 = 0;
            while(i1 < this.f00[i].length)
            {
                i0 = i0 + this.f00[i][i1];
                if (!(this.f00[i][i1] != 0))
                {
                    b = true;
                }
                i1 = i1 + 1;
            }
            if (!b && i0 != this.f30)
            {
                return false;
            }
            i = i + 1;
        }
        return true;
    }
    
    public boolean m20()
    {
        int i = 0;
        while(i < this.f00[0].length)
        {
            int i0 = 0;
            boolean b = false;
            int i1 = 0;
            while(i1 < this.f00.length)
            {
                i0 = i0 + this.f00[i1][i];
                if (!(this.f00[i1][i] != 0))
                {
                    b = true;
                }
                i1 = i1 + 1;
            }
            if (!b && i0 != this.f30)
            {
                return false;
            }
            i = i + 1;
        }
        return true;
    }
    
    public boolean m30()
    {
        int i = 0;
        boolean b = false;
        int i0 = 0;
        while(i0 < this.f00.length)
        {
            i = i + this.f00[i0][i0];
            if (!(this.f00[i0][i0] != 0))
            {
                b = true;
            }
            i0 = i0 + 1;
        }
        if (!b && i != this.f30)
        {
            return false;
        }
        int i1 = 0;
        boolean b0 = false;
        int i2 = 0;
        while(i2 < this.f00.length)
        {
            i1 = i1 + this.f00[i2][this.f00.length - 1 - i2];
            if (!(this.f00[i2][this.f00.length - 1 - i2] != 0))
            {
                b0 = true;
            }
            i2 = i2 + 1;
        }
        if (!b0 && i1 != this.f30)
        {
            return false;
        }
        return true;
    }
    
    public int m40()
    {
        return this.f40;
    }
    
    public String toString()
    {
        String s = "";
        int i = 0;
        while(i < this.f00.length)
        {
            int i0 = 0;
            while(i0 < this.f00[i].length)
            {
                s = new StringBuilder().append(s).append(0 + this.f00[i][i0]).append("\t").toString();
                i0 = i0 + 1;
            }
            s = new StringBuilder().append(s).append("0\n").toString();
            i = i + 1;
        }
        return s;
    }
}
