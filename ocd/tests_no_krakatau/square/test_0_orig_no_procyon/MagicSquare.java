public class MagicSquare {
    private int[][] square;
    private boolean[] possible;
    private int totalSqs;
    private int sum;
    private int numsquares;
    
    public static void main(String[] a)
    {
        MagicSquare a0 = new MagicSquare(3);
        a0.fill(0, 0);
        System.out.println(new StringBuilder().append("There were ").append(a0.getTotalSolutions()).append(" number of solutions.").toString());
    }
    
    public MagicSquare(int i)
    {
        super();
        this.square = new int[i][i];
        int i0 = 0;
        while(i0 < i)
        {
            int i1 = 0;
            while(i1 < i)
            {
                this.square[i0][i1] = 0;
                i1 = i1 + 1;
            }
            i0 = i0 + 1;
        }
        this.totalSqs = i * i;
        this.possible = new boolean[this.totalSqs];
        int i2 = 0;
        while(i2 < this.totalSqs)
        {
            this.possible[i2] = true;
            i2 = i2 + 1;
        }
        this.sum = i * (i * i + 1) / 2;
        this.numsquares = 0;
    }
    
    public void fill(int i, int i0)
    {
        if (this.checkRows() && this.checkCols() && this.checkDiags())
        {
            if (i == this.square.length)
            {
                System.out.println((Object)this);
                this.numsquares = this.numsquares + 1;
                return;
            }
            int i1 = 0;
            while(i1 < this.totalSqs)
            {
                if (this.possible[i1])
                {
                    int i2 = 0;
                    this.square[i][i0] = i1 + 1;
                    this.possible[i1] = false;
                    int i3 = i0 + 1;
                    if (i3 != this.square.length)
                    {
                        i2 = i;
                    }
                    else
                    {
                        i2 = i + 1;
                        i3 = 0;
                    }
                    this.fill(i2, i3);
                    this.square[i][i0] = 0;
                    this.possible[i1] = true;
                }
                i1 = i1 + 1;
            }
            return;
        }
    }
    
    public boolean checkRows()
    {
        int i = 0;
        while(i < this.square.length)
        {
            int i0 = 0;
            boolean b = false;
            int i1 = 0;
            while(i1 < this.square[i].length)
            {
                i0 = i0 + this.square[i][i1];
                if (this.square[i][i1] == 0)
                {
                    b = true;
                }
                i1 = i1 + 1;
            }
            if (!b && i0 != this.sum)
            {
                return false;
            }
            i = i + 1;
        }
        return true;
    }
    
    public boolean checkCols()
    {
        int i = 0;
        while(i < this.square[0].length)
        {
            int i0 = 0;
            boolean b = false;
            int i1 = 0;
            while(i1 < this.square.length)
            {
                i0 = i0 + this.square[i1][i];
                if (this.square[i1][i] == 0)
                {
                    b = true;
                }
                i1 = i1 + 1;
            }
            if (!b && i0 != this.sum)
            {
                return false;
            }
            i = i + 1;
        }
        return true;
    }
    
    public boolean checkDiags()
    {
        boolean b = false;
        int i = 0;
        boolean b0 = false;
        int i0 = 0;
        while(i0 < this.square.length)
        {
            i = i + this.square[i0][i0];
            if (this.square[i0][i0] == 0)
            {
                b0 = true;
            }
            i0 = i0 + 1;
        }
        if (!b0 && i != this.sum)
        {
            return false;
        }
        int i1 = 0;
        boolean b1 = false;
        int i2 = 0;
        while(i2 < this.square.length)
        {
            i1 = i1 + this.square[i2][this.square.length - 1 - i2];
            if (this.square[i2][this.square.length - 1 - i2] == 0)
            {
                b1 = true;
            }
            i2 = i2 + 1;
        }
        label2: {
            label0: {
                label1: {
                    if (b1)
                    {
                        break label1;
                    }
                    if (i1 != this.sum)
                    {
                        break label0;
                    }
                }
                b = true;
                break label2;
            }
            b = false;
        }
        return b;
    }
    
    public int getTotalSolutions()
    {
        return this.numsquares;
    }
    
    public String toString()
    {
        String s = "";
        int i = 0;
        while(i < this.square.length)
        {
            int i0 = 0;
            while(i0 < this.square[i].length)
            {
                s = new StringBuilder().append(s).append((this.square[i][i0] != 0) ? 1 : 0).append("\t").toString();
                i0 = i0 + 1;
            }
            s = new StringBuilder().append(s).append("\n").toString();
            i = i + 1;
        }
        return s;
    }
}
